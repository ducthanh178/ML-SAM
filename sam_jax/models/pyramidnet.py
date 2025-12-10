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

"""PyramidNet model with ShakeDrop regularization - Flax Linen version."""

from typing import Tuple
from flax import linen as nn
import jax
import jax.numpy as jnp
from sam_new.sam_jax.models import utils


def _shortcut(x: jnp.ndarray, chn_out: int, strides: Tuple[int, int]) -> jnp.ndarray:
  """Pyramid Net shortcut."""
  chn_in = x.shape[3]
  if strides != (1, 1):
    x = nn.avg_pool(x, strides, strides)
  if chn_out != chn_in:
    diff = chn_out - chn_in
    x = jnp.pad(x, [[0, 0], [0, 0], [0, 0], [0, diff]])
  return x


class BottleneckShakeDrop(nn.Module):
  """PyramidNet with Shake-Drop Bottleneck - Flax Linen version."""
  
  channels: int
  strides: Tuple[int, int] = (1, 1)
  prob: float = 0.5
  alpha_min: float = -1.0
  alpha_max: float = 1.0
  beta_min: float = 0.0
  beta_max: float = 1.0
  true_gradient: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
    """Forward pass."""
    # BN + ReLU (optional)
    bn1_pre = nn.BatchNorm(use_running_average=not train, name='bn_1_pre')
    y = bn1_pre(x)
    
    # 1x1 conv contract
    conv1 = nn.Conv(
        features=self.channels,
        kernel_size=(1, 1),
        padding='SAME',
        use_bias=False,
        kernel_init=utils.conv_kernel_init_fn,
        name='1x1_conv_contract')
    y = conv1(y)
    
    bn1_post = nn.BatchNorm(use_running_average=not train, name='bn_1_post')
    y = bn1_post(y)
    y = nn.relu(y)
    
    # 3x3 conv
    conv2 = nn.Conv(
        features=self.channels,
        kernel_size=(3, 3),
        strides=self.strides,
        padding='SAME',
        use_bias=False,
        kernel_init=utils.conv_kernel_init_fn,
        name='3x3')
    y = conv2(y)
    
    bn2 = nn.BatchNorm(use_running_average=not train, name='bn_2')
    y = bn2(y)
    y = nn.relu(y)
    
    # 1x1 conv expand
    conv3 = nn.Conv(
        features=self.channels * 4,
        kernel_size=(1, 1),
        padding='SAME',
        use_bias=False,
        kernel_init=utils.conv_kernel_init_fn,
        name='1x1_conv_expand')
    y = conv3(y)
    
    bn3 = nn.BatchNorm(use_running_average=not train, name='bn_3')
    y = bn3(y)
    
    # ShakeDrop
    if train:
      # In Linen, we need to get RNG from the module
      rng = self.make_rng('dropout')
      y = utils.shake_drop_train(
          y, self.prob, self.alpha_min, self.alpha_max,
          self.beta_min, self.beta_max, rng=rng)
    else:
      y = utils.shake_drop_eval(y, self.prob, self.alpha_min, self.alpha_max)
    
    x = _shortcut(x, self.channels * 4, self.strides)
    return x + y


def _calc_shakedrop_mask_prob(curr_layer: int, total_layers: int, mask_prob: float) -> float:
  """Calculates drop prob depending on the current layer."""
  return 1 - (float(curr_layer) / total_layers) * mask_prob


class PyramidNetShakeDrop(nn.Module):
  """PyramidNet with Shake-Drop - Flax Linen version."""
  
  num_outputs: int
  pyramid_alpha: int = 200
  pyramid_depth: int = 272
  true_gradient: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
    """Forward pass."""
    assert (self.pyramid_depth - 2) % 9 == 0
    
    # Shake-drop hyper-params
    mask_prob = 0.5
    alpha_min, alpha_max = (-1.0, 1.0)
    beta_min, beta_max = (0.0, 1.0)
    
    # Bottleneck network size
    blocks_per_group = (self.pyramid_depth - 2) // 9
    num_channels = 16
    total_blocks = blocks_per_group * 3
    delta_channels = self.pyramid_alpha / total_blocks
    
    # Initial conv
    conv_init = nn.Conv(
        features=16,
        kernel_size=(3, 3),
        padding='SAME',
        name='init_conv',
        use_bias=False,
        kernel_init=utils.conv_kernel_init_fn)
    x = conv_init(x)
    
    bn_init = nn.BatchNorm(use_running_average=not train, name='init_bn')
    x = bn_init(x)
    
    layer_num = 1
    
    # Group 1
    for block_i in range(blocks_per_group):
      num_channels += delta_channels
      layer_mask_prob = _calc_shakedrop_mask_prob(layer_num, total_blocks, mask_prob)
      x = BottleneckShakeDrop(
          channels=int(round(num_channels)),
          strides=(1, 1),
          prob=layer_mask_prob,
          alpha_min=alpha_min,
          alpha_max=alpha_max,
          beta_min=beta_min,
          beta_max=beta_max,
          true_gradient=self.true_gradient,
          name=f'block_1_{block_i}')(x, train=train)
      layer_num += 1
    
    # Group 2
    for block_i in range(blocks_per_group):
      num_channels += delta_channels
      layer_mask_prob = _calc_shakedrop_mask_prob(layer_num, total_blocks, mask_prob)
      x = BottleneckShakeDrop(
          channels=int(round(num_channels)),
          strides=((2, 2) if block_i == 0 else (1, 1)),
          prob=layer_mask_prob,
          alpha_min=alpha_min,
          alpha_max=alpha_max,
          beta_min=beta_min,
          beta_max=beta_max,
          true_gradient=self.true_gradient,
          name=f'block_2_{block_i}')(x, train=train)
      layer_num += 1
    
    # Group 3
    for block_i in range(blocks_per_group):
      num_channels += delta_channels
      layer_mask_prob = _calc_shakedrop_mask_prob(layer_num, total_blocks, mask_prob)
      x = BottleneckShakeDrop(
          channels=int(round(num_channels)),
          strides=((2, 2) if block_i == 0 else (1, 1)),
          prob=layer_mask_prob,
          alpha_min=alpha_min,
          alpha_max=alpha_max,
          beta_min=beta_min,
          beta_max=beta_max,
          true_gradient=self.true_gradient,
          name=f'block_3_{block_i}')(x, train=train)
      layer_num += 1
    
    assert layer_num - 1 == total_blocks
    
    # Final layers
    bn_final = nn.BatchNorm(use_running_average=not train, name='final_bn')
    x = bn_final(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, (8, 8))
    x = x.reshape((x.shape[0], -1))
    
    dense = nn.Dense(
        features=self.num_outputs,
        kernel_init=utils.dense_layer_init_fn,
        name='dense')
    x = dense(x)
    return x

