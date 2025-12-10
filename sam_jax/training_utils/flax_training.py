# Copyright 2020 The Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions to train the networks for image classification - Flax Linen + Optax version."""

import dataclasses
import functools
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from absl import flags
from absl import logging
from flax import linen as nn
from flax import jax_utils
from flax.training import checkpoints
import optax
import jax
import jax.numpy as jnp
import numpy as np
from sam_new.sam_jax.datasets import dataset_source as dataset_source_lib
import tensorflow as tf
from tensorflow.io import gfile

FLAGS = flags.FLAGS

# Training hyper-parameters
flags.DEFINE_float('gradient_clipping', 5.0, 'Gradient clipping.')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_bool('use_learning_rate_schedule', True,
                  'Whether to use a cosine schedule or keep the learning rate constant.')
flags.DEFINE_float('weight_decay', 0.001, 'Weight decay coefficient.')
flags.DEFINE_integer('run_seed', 0, 'Seed for random number generation.')
flags.DEFINE_bool('use_rmsprop', False, 'If True, uses RMSprop instead of SGD')
flags.DEFINE_enum('lr_schedule', 'cosine', ['cosine', 'exponential'],
                  'Learning rate schedule to use.')

# Additional flags
flags.DEFINE_integer('save_progress_seconds', 3600, 'Save progress every...s')
flags.DEFINE_multi_integer('additional_checkpoints_at_epochs', [],
                           'Additional epochs when we should save the model.')
flags.DEFINE_bool('also_eval_on_training_set', False,
                  'If set to true, the model will also be evaluated on the training set.')
flags.DEFINE_bool('compute_top_5_error_rate', False,
                  'If true, will also compute top 5 error rate.')
flags.DEFINE_float('label_smoothing', 0.0, 'Label smoothing for cross entropy.')
flags.DEFINE_float('ema_decay', 0.0, 'If not zero, use EMA on all weights.')
flags.DEFINE_bool('no_weight_decay_on_bn', False,
                  'If set to True, will not apply weight decay on the batch norm parameters.')
flags.DEFINE_integer('evaluate_every', 1, 'Evaluate on the test set every n epochs.')

# SAM related flags
flags.DEFINE_float('sam_rho', 0.0,
                   'Size of the neighborhood considered for the SAM perturbation.')
flags.DEFINE_bool('sync_perturbations', False,
                  'If set to True, sync the adversarial perturbation between replicas.')
flags.DEFINE_integer('inner_group_size', None,
                     'Inner group size for syncing the adversarial gradients.')


def cross_entropy_loss(logits: jnp.ndarray,
                       one_hot_labels: jnp.ndarray,
                       mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Returns the cross entropy loss."""
  if FLAGS.label_smoothing > 0:
    smoothing = jnp.ones_like(one_hot_labels) / one_hot_labels.shape[-1]
    one_hot_labels = ((1 - FLAGS.label_smoothing) * one_hot_labels +
                      FLAGS.label_smoothing * smoothing)
  log_softmax_logits = jax.nn.log_softmax(logits)
  if mask is None:
    mask = jnp.ones([logits.shape[0]])
  mask = mask.reshape([logits.shape[0], 1])
  loss = -jnp.sum(one_hot_labels * log_softmax_logits * mask) / mask.sum()
  return jnp.nan_to_num(loss)


def error_rate_metric(logits: jnp.ndarray,
                      one_hot_labels: jnp.ndarray,
                      mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Returns the error rate."""
  if mask is None:
    mask = jnp.ones([logits.shape[0]])
  mask = mask.reshape([logits.shape[0]])
  error_rate = (((jnp.argmax(logits, -1) != jnp.argmax(one_hot_labels, -1))) *
                mask).sum() / mask.sum()
  return jnp.nan_to_num(error_rate)


def top_k_error_rate_metric(logits: jnp.ndarray,
                            one_hot_labels: jnp.ndarray,
                            k: int = 5,
                            mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Returns the top-K error rate."""
  if mask is None:
    mask = jnp.ones([logits.shape[0]])
  mask = mask.reshape([logits.shape[0]])
  true_labels = jnp.argmax(one_hot_labels, -1).reshape([-1, 1])
  top_k_preds = jnp.argsort(logits, axis=-1)[:, -k:]
  hit = jax.vmap(jnp.isin)(true_labels, top_k_preds)
  error_rate = 1 - ((hit * mask).sum() / mask.sum())
  return jnp.nan_to_num(error_rate)


def get_cosine_schedule(num_epochs: int, learning_rate: float,
                        num_training_obs: int,
                        batch_size: int) -> Callable[[int], float]:
  """Returns a cosine learning rate schedule."""
  steps_per_epoch = int(math.floor(num_training_obs / batch_size))
  total_steps = steps_per_epoch * num_epochs
  
  def learning_rate_fn(step):
    progress = step / total_steps
    return learning_rate * 0.5 * (1 + jnp.cos(jnp.pi * progress))
  
  return learning_rate_fn


def get_exponential_schedule(num_epochs: int, learning_rate: float,
                              num_training_obs: int,
                              batch_size: int) -> Callable[[int], float]:
  """Returns an exponential learning rate schedule."""
  steps_per_epoch = int(math.floor(num_training_obs / batch_size))
  end_lr_ratio = 0.012
  lamba = -num_epochs / math.log(end_lr_ratio)
  
  def learning_rate_fn(step):
    t = step / steps_per_epoch
    return learning_rate * jnp.exp(-t / lamba)
  
  return learning_rate_fn


def global_norm(updates) -> jnp.ndarray:
  """Returns the l2 norm of the input."""
  return jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(updates)]))


def clip_by_global_norm(updates, clip_norm: float):
  """Clips the gradient by global norm."""
  if clip_norm > 0:
    g_norm = global_norm(updates)
    trigger = g_norm < clip_norm
    updates = jax.tree_map(
        lambda t: jnp.where(trigger, t, (t / g_norm) * clip_norm),
        updates)
  return updates


def dual_vector(y) -> jnp.ndarray:
  """Returns the solution of max_x y^T x s.t. ||x||_2 <= 1."""
  gradient_norm = jnp.sqrt(sum(
      [jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(y)]))
  normalized_gradient = jax.tree_map(lambda x: x / gradient_norm, y)
  return normalized_gradient


def create_optimizer(learning_rate_fn: Callable[[int], float],
                     weight_decay: float = 0.001,
                     beta: float = 0.9) -> optax.GradientTransformation:
  """Creates an optimizer using Optax."""
  if FLAGS.use_rmsprop:
    # RMSprop with momentum - using optax's RMSprop
    optimizer = optax.rmsprop(
        learning_rate=learning_rate_fn,
        momentum=beta,
        eps=0.001,
        decay=0.9)
  else:
    # SGD with Nesterov momentum
    optimizer = optax.sgd(
        learning_rate=learning_rate_fn,
        momentum=beta,
        nesterov=True)
  
  # Add weight decay
  if weight_decay > 0:
    optimizer = optax.chain(
        optimizer,
        optax.add_decayed_weights(weight_decay))
  
  return optimizer


@dataclasses.dataclass
class TrainState:
  """Training state with batch statistics."""
  step: int
  apply_fn: Callable
  params: Dict[str, Any]
  tx: optax.GradientTransformation
  opt_state: optax.OptState
  batch_stats: Dict[str, Any]
  
  @classmethod
  def create(cls, *, apply_fn, params, tx, opt_state, batch_stats, **kwargs):
    return cls(
        step=0,
        apply_fn=apply_fn,
        params=params,
        tx=tx,
        opt_state=opt_state,
        batch_stats=batch_stats,
        **kwargs)
  
  def replace(self, **kwargs):
    return dataclasses.replace(self, **kwargs)


def train_step(state: TrainState,
               batch: Dict[str, jnp.ndarray],
               prng_key: jnp.ndarray,
               learning_rate_fn: Callable[[int], float],
               l2_reg: float,
               model: nn.Module,
               rho: float = 0.0) -> Tuple[TrainState, Dict[str, float], float]:
  """Performs one gradient step with optional SAM."""
  
  def forward_and_loss(params, batch_stats, rng, true_gradient=False):
    """Forward pass and loss computation."""
    variables = {'params': params, 'batch_stats': batch_stats}
    logits, new_batch_stats = model.apply(
        variables, batch['image'], train=True, mutable=['batch_stats'], rngs={'dropout': rng})
    loss = cross_entropy_loss(logits, batch['label'])
    
    # Weight decay
    weight_penalty_params = jax.tree_util.tree_leaves(params)
    if FLAGS.no_weight_decay_on_bn:
      weight_l2 = sum([jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1])
    else:
      weight_l2 = sum([jnp.sum(x**2) for x in weight_penalty_params])
    weight_penalty = l2_reg * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, (new_batch_stats, logits)
  
  step = state.step
  
  def get_sam_gradient(params, batch_stats, rng, rho_val):
    """Returns gradient using SAM perturbation."""
    # Compute gradient at current params
    (_, _), grad = jax.value_and_grad(
        lambda p: forward_and_loss(p, batch_stats, rng, true_gradient=True)[0],
        has_aux=False)(params)
    
    if FLAGS.sync_perturbations:
      grad = jax.lax.pmean(grad, 'batch')
    
    # Normalize gradient
    grad = dual_vector(grad)
    
    # Create perturbed params
    perturbed_params = jax.tree_map(lambda a, b: a + rho_val * b, params, grad)
    
    # Compute gradient at perturbed params
    (_, (new_batch_stats, logits)), grad = jax.value_and_grad(
        lambda p: forward_and_loss(p, batch_stats, rng)[0],
        has_aux=True)(perturbed_params)
    
    return (new_batch_stats, logits), grad
  
  lr = learning_rate_fn(step)
  
  if rho > 0:  # SAM
    (new_batch_stats, logits), grad = get_sam_gradient(
        state.params, state.batch_stats, prng_key, rho)
  else:  # Standard SGD
    (_, (new_batch_stats, logits)), grad = jax.value_and_grad(
        lambda p: forward_and_loss(p, state.batch_stats, prng_key)[0],
        has_aux=True)(state.params)
  
  # Synchronize gradients
  grad = jax.lax.pmean(grad, 'batch')
  
  # Gradient clipping
  grad = clip_by_global_norm(grad, FLAGS.gradient_clipping)
  
  # Update parameters
  updates, new_opt_state = state.tx.update(grad, state.opt_state, state.params)
  new_params = optax.apply_updates(state.params, updates)
  
  new_state = state.replace(
      step=step + 1,
      params=new_params,
      opt_state=new_opt_state,
      batch_stats=new_batch_stats)
  
  # Compute metrics
  gradient_norm = global_norm(grad)
  param_norm = global_norm(new_params)
  
  metrics = {
      'train_error_rate': error_rate_metric(logits, batch['label']),
      'train_loss': cross_entropy_loss(logits, batch['label']),
      'gradient_norm': gradient_norm,
      'param_norm': param_norm}
  
  return new_state, metrics, lr


def eval_step(state: TrainState,
              batch: Dict[str, jnp.ndarray],
              model: nn.Module) -> Dict[str, float]:
  """Evaluates the model on a single batch."""
  # Average batch stats across replicas
  batch_stats = jax.lax.pmean(state.batch_stats, 'batch')
  
  variables = {'params': state.params, 'batch_stats': batch_stats}
  logits = model.apply(variables, batch['image'], train=False)
  
  num_samples = (batch['image'].shape[0] if 'mask' not in batch
                 else batch['mask'].sum())
  mask = batch.get('mask', None)
  labels = batch['label']
  
  metrics = {
      'error_rate': error_rate_metric(logits, labels, mask) * num_samples,
      'loss': cross_entropy_loss(logits, labels, mask) * num_samples}
  
  if FLAGS.compute_top_5_error_rate:
    metrics['top_5_error_rate'] = (
        top_k_error_rate_metric(logits, labels, 5, mask) * num_samples)
  
  metrics = jax.lax.psum(metrics, 'batch')
  return metrics


def tensorflow_to_numpy(xs):
  """Converts tensorflow tensors to numpy arrays."""
  return jax.tree_map(lambda x: x._numpy(), xs)  # pylint: disable=protected-access


def shard_batch(xs):
  """Shards a batch across all available replicas."""
  local_device_count = jax.local_device_count()
  def _prepare(x):
    return x.reshape((local_device_count, -1) + x.shape[1:])
  return jax.tree_map(_prepare, xs)


def load_and_shard_tf_batch(xs):
  """Converts to numpy arrays and distribute a tensorflow batch."""
  xs = tensorflow_to_numpy(xs)
  return shard_batch(xs)


def create_train_state(model: nn.Module,
                       params: Dict[str, Any],
                       batch_stats: Dict[str, Any],
                       learning_rate_fn: Callable[[int], float],
                       weight_decay: float = 0.001) -> TrainState:
  """Creates initial training state."""
  tx = create_optimizer(learning_rate_fn, weight_decay)
  opt_state = tx.init(params)
  
  return TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=tx,
      opt_state=opt_state,
      batch_stats=batch_stats)


def save_checkpoint(state: TrainState,
                    directory: str,
                    epoch: int):
  """Saves a checkpoint."""
  if jax.host_id() != 0:
    return
  # Sync across replicas before saving
  unreplicated_state = jax_utils.unreplicate(state)
  ckpt = {
      'step': unreplicated_state.step,
      'params': unreplicated_state.params,
      'opt_state': unreplicated_state.opt_state,
      'batch_stats': unreplicated_state.batch_stats,
      'epoch': epoch
  }
  checkpoints.save_checkpoint(directory, ckpt, epoch, keep=2, overwrite=True)


def restore_checkpoint(state: TrainState,
                       directory: str) -> Tuple[TrainState, int]:
  """Restores a checkpoint."""
  ckpt = {
      'step': state.step,
      'params': state.params,
      'opt_state': state.opt_state,
      'batch_stats': state.batch_stats,
      'epoch': 0
  }
  restored_ckpt = checkpoints.restore_checkpoint(directory, ckpt)
  restored_state = state.replace(
      step=restored_ckpt['step'],
      params=restored_ckpt['params'],
      opt_state=restored_ckpt['opt_state'],
      batch_stats=restored_ckpt['batch_stats'])
  return restored_state, restored_ckpt['epoch']


class ExponentialMovingAverage:
  """Exponential Moving Average for parameters."""
  
  def __init__(self, params: Dict[str, Any], batch_stats: Dict[str, Any],
               decay: float, warmup_steps: int):
    self.param_ema = (params, batch_stats)
    self.decay = decay
    self.warmup_steps = warmup_steps
  
  def update(self, new_params: Dict[str, Any], new_batch_stats: Dict[str, Any],
             step: int):
    """Updates the moving average."""
    factor = float(step >= self.warmup_steps)
    delta = step - self.warmup_steps
    decay = min(self.decay, (1. + delta) / (10. + delta))
    decay *= factor
    
    def update_ema(a, b):
      return (1 - decay) * a + decay * b
    
    ema_params, ema_batch_stats = self.param_ema
    new_ema_params = jax.tree_map(update_ema, ema_params, new_params)
    new_ema_batch_stats = jax.tree_map(update_ema, ema_batch_stats, new_batch_stats)
    self.param_ema = (new_ema_params, new_ema_batch_stats)
    return self


def train(model: nn.Module,
          params: Dict[str, Any],
          batch_stats: Dict[str, Any],
          dataset_source: dataset_source_lib.DatasetSource,
          training_dir: str,
          num_epochs: int):
  """Trains the model."""
  checkpoint_dir = os.path.join(training_dir, 'checkpoints')
  if not gfile.exists(checkpoint_dir):
    gfile.makedirs(checkpoint_dir)
  
  prng_key = jax.random.PRNGKey(FLAGS.run_seed)
  
  # Learning rate schedule
  if FLAGS.use_learning_rate_schedule:
    if FLAGS.lr_schedule == 'cosine':
      learning_rate_fn = get_cosine_schedule(
          num_epochs, FLAGS.learning_rate,
          dataset_source.num_training_obs,
          dataset_source.batch_size)
    elif FLAGS.lr_schedule == 'exponential':
      learning_rate_fn = get_exponential_schedule(
          num_epochs, FLAGS.learning_rate,
          dataset_source.num_training_obs,
          dataset_source.batch_size)
    else:
      raise ValueError('Wrong schedule: ' + FLAGS.lr_schedule)
  else:
    learning_rate_fn = lambda step: FLAGS.learning_rate
  
  # Create training state
  state = create_train_state(
      model, params, batch_stats, learning_rate_fn, FLAGS.weight_decay)
  
  # Restore checkpoint if exists
  initial_epoch = 0
  if gfile.exists(checkpoint_dir) and gfile.listdir(checkpoint_dir):
    try:
      state, initial_epoch = restore_checkpoint(state, checkpoint_dir)
      initial_epoch += 1
      logging.info(f'Resuming training from epoch {initial_epoch}')
    except Exception as e:
      logging.warning(f'Could not restore checkpoint: {e}')
  
  # EMA
  moving_averages = None
  if FLAGS.ema_decay > 0:
    end_warmup_step = 1560
    moving_averages = ExponentialMovingAverage(
        params, batch_stats, FLAGS.ema_decay, end_warmup_step)
  
  # Replicate state
  state = jax_utils.replicate(state)
  if moving_averages:
    moving_averages.param_ema = jax_utils.replicate(moving_averages.param_ema)
  
  # PMap training and evaluation
  pmapped_train_step = jax.pmap(
      functools.partial(
          train_step,
          learning_rate_fn=learning_rate_fn,
          l2_reg=FLAGS.weight_decay,
          model=model,
          rho=FLAGS.sam_rho),
      axis_name='batch',
      donate_argnums=(0,))
  
  pmapped_eval_step = jax.pmap(
      functools.partial(eval_step, model=model),
      axis_name='batch')
  
  time_at_last_checkpoint = time.time()
  
  # Training loop
  for epoch in range(initial_epoch, num_epochs):
    logging.info(f'Starting epoch {epoch}')
    start_time = time.time()
    
    # Training
    train_metrics = []
    for batch in dataset_source.get_train(use_augmentations=True):
      batch = load_and_shard_tf_batch(batch)
      step_key = jax.random.fold_in(prng_key, state.step[0])
      sharded_keys = jax_utils.replicate(step_key)
      
      state, metrics, lr = pmapped_train_step(state, batch, sharded_keys)
      
      # Update EMA
      if moving_averages:
        unreplicated_state = jax_utils.unreplicate(state)
        moving_averages.update(
            unreplicated_state.params,
            unreplicated_state.batch_stats,
            int(unreplicated_state.step))
      
      train_metrics.append(metrics)
    
    # Log training metrics
    train_metrics = jax_utils.unreplicate(train_metrics)
    avg_metrics = jax.tree_map(lambda x: x.mean(), train_metrics)
    logging.info(f'Epoch {epoch} training metrics: {avg_metrics}')
    
    # Evaluation
    if (epoch + 1) % FLAGS.evaluate_every == 0:
      test_metrics = []
      for batch in dataset_source.get_test():
        batch = load_and_shard_tf_batch(batch)
        metrics = pmapped_eval_step(state, batch)
        test_metrics.append(metrics)
      
      test_metrics = jax_utils.unreplicate(test_metrics)
      total_error = sum(m['error_rate'] for m in test_metrics)
      total_loss = sum(m['loss'] for m in test_metrics)
      total_samples = sum(batch['image'].shape[0] for batch in dataset_source.get_test())
      avg_error = total_error / total_samples
      avg_loss = total_loss / total_samples
      logging.info(f'Epoch {epoch} test error: {avg_error:.4f}, loss: {avg_loss:.4f}')
      
      # Evaluate with EMA if available
      if moving_averages:
        ema_params, ema_batch_stats = moving_averages.param_ema
        ema_state = state.replace(params=ema_params, batch_stats=ema_batch_stats)
        ema_test_metrics = []
        for batch in dataset_source.get_test():
          batch = load_and_shard_tf_batch(batch)
          metrics = pmapped_eval_step(ema_state, batch)
          ema_test_metrics.append(metrics)
        ema_test_metrics = jax_utils.unreplicate(ema_test_metrics)
        ema_total_error = sum(m['error_rate'] for m in ema_test_metrics)
        ema_avg_error = ema_total_error / total_samples
        logging.info(f'Epoch {epoch} EMA test error: {ema_avg_error:.4f}')
    
    # Save checkpoint
    sec_from_last_ckpt = time.time() - time_at_last_checkpoint
    if sec_from_last_ckpt > FLAGS.save_progress_seconds:
      save_checkpoint(state, checkpoint_dir, epoch)
      time_at_last_checkpoint = time.time()
      logging.info('Saved checkpoint.')
    
    elapsed = time.time() - start_time
    logging.info(f'Epoch {epoch} finished in {elapsed:.2f}s')
  
  # Save final checkpoint
  save_checkpoint(state, checkpoint_dir, num_epochs - 1)

