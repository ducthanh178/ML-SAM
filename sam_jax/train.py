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

"""Trains a model on cifar10, cifar100, SVHN, F-MNIST or imagenet - Flax Linen + Optax version."""

import os

from absl import app
from absl import flags
from absl import logging
import jax
from sam_new.sam_jax.datasets import dataset_source as dataset_source_lib
from sam_new.sam_jax.datasets import dataset_source_imagenet
from sam_new.sam_jax.imagenet_models import load_model as load_imagenet_model
from sam_new.sam_jax.models import load_model
from sam_new.sam_jax.training_utils import flax_training
import tensorflow.compat.v2 as tf
from tensorflow.io import gfile


FLAGS = flags.FLAGS

flags.DEFINE_enum('dataset', 'cifar10', [
    'cifar10', 'cifar100', 'fashion_mnist', 'svhn', 'imagenet'
], 'Name of the dataset.')
flags.DEFINE_enum('model_name', 'WideResnet28x10', [
    'WideResnet28x10', 'WideResnet28x6_ShakeShake', 'Pyramid_ShakeDrop',
    'WideResnet_mini', 'WideResnet_ShakeShake_mini', 'Pyramid_ShakeDrop_mini',
    'Resnet50', 'Resnet101', 'Resnet152'
], 'Name of the model to train.')
flags.DEFINE_integer('num_epochs', 200,
                     'How many epochs the model should be trained for.')
flags.DEFINE_integer(
    'batch_size', 128, 'Global batch size. If multiple '
    'replicas are used, each replica will receive '
    'batch_size / num_replicas examples. Batch size should be divisible by '
    'the number of available devices.')
flags.DEFINE_string(
    'output_dir', '', 'Directory where the checkpoints and the tensorboard '
    'records should be saved.')
flags.DEFINE_enum(
    'image_level_augmentations', 'basic', ['none', 'basic', 'autoaugment',
                                           'aa-only'],
    'Augmentations applied to the images.')
flags.DEFINE_enum(
    'batch_level_augmentations', 'none', ['none', 'cutout', 'mixup', 'mixcut'],
    'Augmentations that are applied at the batch level.')


def main(_):
  tf.enable_v2_behavior()
  # make sure tf does not allocate gpu memory
  tf.config.experimental.set_visible_devices([], 'GPU')

  # Performance gains on TPU by switching to hardware bernoulli.
  def hardware_bernoulli(rng_key, p=jax.numpy.float32(0.5), shape=None):
    lax_key = jax.lax.tie_in(rng_key, 0.0)
    return jax.lax.rng_uniform(lax_key, 1.0, shape) < p

  def set_hardware_bernoulli():
    jax.random.bernoulli = hardware_bernoulli

  set_hardware_bernoulli()

  # Output directory
  output_dir_suffix = os.path.join(
      'lr_' + str(FLAGS.learning_rate),
      'wd_' + str(FLAGS.weight_decay),
      'rho_' + str(FLAGS.sam_rho),
      'seed_' + str(FLAGS.run_seed))

  output_dir = os.path.join(FLAGS.output_dir, output_dir_suffix)

  if not gfile.exists(output_dir):
    gfile.makedirs(output_dir)

  num_devices = jax.local_device_count() * jax.host_count()
  assert FLAGS.batch_size % num_devices == 0
  local_batch_size = FLAGS.batch_size // num_devices
  info = 'Total batch size: {} ({} x {} replicas)'.format(
      FLAGS.batch_size, local_batch_size, num_devices)
  logging.info(info)

  # Create dataset
  if FLAGS.dataset == 'cifar10':
    image_size = 32
    dataset_source = dataset_source_lib.Cifar10(
        FLAGS.batch_size // jax.host_count(),
        FLAGS.image_level_augmentations,
        FLAGS.batch_level_augmentations,
        image_size=image_size)
  elif FLAGS.dataset == 'cifar100':
    image_size = 32
    dataset_source = dataset_source_lib.Cifar100(
        FLAGS.batch_size // jax.host_count(),
        FLAGS.image_level_augmentations,
        FLAGS.batch_level_augmentations,
        image_size=image_size)
  elif FLAGS.dataset == 'fashion_mnist':
    image_size = 28
    dataset_source = dataset_source_lib.FashionMnist(
        FLAGS.batch_size, FLAGS.image_level_augmentations,
        FLAGS.batch_level_augmentations)
  elif FLAGS.dataset == 'svhn':
    image_size = 32
    dataset_source = dataset_source_lib.SVHN(
        FLAGS.batch_size, FLAGS.image_level_augmentations,
        FLAGS.batch_level_augmentations)
  elif FLAGS.dataset == 'imagenet':
    image_size = 224  # Default for ResNet
    dataset_source = dataset_source_imagenet.Imagenet(
        FLAGS.batch_size // jax.host_count(), image_size,
        FLAGS.image_level_augmentations)
  else:
    raise ValueError('Dataset not recognized.')

  # Determine num_classes and num_channels
  if 'cifar' in FLAGS.dataset:
    num_channels = 3
    num_classes = 100 if FLAGS.dataset == 'cifar100' else 10
  elif FLAGS.dataset == 'fashion_mnist':
    num_channels = 1
    num_classes = 10
  elif FLAGS.dataset == 'svhn':
    num_channels = 3
    num_classes = 10
  elif FLAGS.dataset == 'imagenet':
    num_channels = 3
    num_classes = 1000
  else:
    raise ValueError('Dataset not recognized.')

  # Load model
  try:
    # Try ImageNet models first
    model, params, batch_stats = load_imagenet_model.get_model(
        FLAGS.model_name, local_batch_size, image_size, num_classes)
  except (ValueError, AttributeError):
    # Fall back to CIFAR models
    model, params, batch_stats = load_model.get_model(
        FLAGS.model_name, local_batch_size, image_size,
        num_classes, num_channels)

  # Train
  flax_training.train(model, params, batch_stats, dataset_source, output_dir,
                      FLAGS.num_epochs)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  app.run(main)

