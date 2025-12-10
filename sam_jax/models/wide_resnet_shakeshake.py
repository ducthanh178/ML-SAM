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

"""Wide Resnet Model with ShakeShake regularization - Flax Linen version."""

from typing import Tuple
from flax import linen as nn
import jax
import jax.numpy as jnp
from sam_new.sam_jax.models import utils


class Shortcut(nn.Module):
  """Shortcut for residual connections - Flax Linen version."""
  
  channels: int
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
    """Forward pass."""
    if x.shape[-1] == self.channels:
      return x
    
    # Skip path 1
    h1 = nn.avg_pool(x, (1, 1), strides=self.strides, padding='VALID')
    conv_h1 = nn.Conv(
        features=self.channels // 2,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='SAME',
        use_bias=False,
        kernel_init=utils.conv_kernel_init_fn,
        name='conv_h1')
    h1 = conv_h1(h1)
    
    # Skip path 2 - offset by one pixel
    pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
    h2 = jnp.pad(x, pad_arr)[:, 1:, 1:, :]
    h2 = nn.avg_pool(h2, (1, 1), strides=self.strides, padding='VALID')
    conv_h2 = nn.Conv(
        features=self.channels // 2,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='SAME',
        use_bias=False,
        kernel_init=utils.conv_kernel_init_fn,
        name='conv_h2')
    h2 = conv_h2(h2)
    
    merged_branches = jnp.concatenate([h1, h2], axis=3)
    
    bn_residual = nn.BatchNorm(use_running_average=not train, name='bn_residual')
    return bn_residual(merged_branches)


class ShakeShakeBlock(nn.Module):
  """Wide ResNet block with shake-shake regularization - Flax Linen version."""
  
  channels: int
  strides: Tuple[int, int] = (1, 1)
  true_gradient: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
    """Forward pass."""
    a = b = residual = x
    
    # Branch A
    a = jax.nn.relu(a)
    conv_a_1 = nn.Conv(
        features=self.channels,
        kernel_size=(3, 3),
        strides=self.strides,
        padding='SAME',
        use_bias=False,
        kernel_init=utils.conv_kernel_init_fn,
        name='conv_a_1')
    a = conv_a_1(a)
    
    bn_a_1 = nn.BatchNorm(use_running_average=not train, name='bn_a_1')
    a = bn_a_1(a)
    a = nn.relu(a)
    
    conv_a_2 = nn.Conv(
        features=self.channels,
        kernel_size=(3, 3),
        padding='SAME',
        use_bias=False,
        kernel_init=utils.conv_kernel_init_fn,
        name='conv_a_2')
    a = conv_a_2(a)
    
    bn_a_2 = nn.BatchNorm(use_running_average=not train, name='bn_a_2')
    a = bn_a_2(a)
    
    # Branch B
    b = jax.nn.relu(b)
    conv_b_1 = nn.Conv(
        features=self.channels,
        kernel_size=(3, 3),
        strides=self.strides,
        padding='SAME',
        use_bias=False,
        kernel_init=utils.conv_kernel_init_fn,
        name='conv_b_1')
    b = conv_b_1(b)
    
    bn_b_1 = nn.BatchNorm(use_running_average=not train, name='bn_b_1')
    b = bn_b_1(b)
    b = nn.relu(b)
    
    conv_b_2 = nn.Conv(
        features=self.channels,
        kernel_size=(3, 3),
        padding='SAME',
        use_bias=False,
        kernel_init=utils.conv_kernel_init_fn,
        name='conv_b_2')
    b = conv_b_2(b)
    
    bn_b_2 = nn.BatchNorm(use_running_average=not train, name='bn_b_2')
    b = bn_b_2(b)
    
    # ShakeShake
    if train:
      rng = self.make_rng('dropout')
      ab = utils.shake_shake_train(a, b, rng=rng)
    else:
      ab = utils.shake_shake_eval(a, b)
    
    # Residual connection
    residual = Shortcut(
        channels=self.channels,
        strides=self.strides,
        name='shortcut')(residual, train=train)
    
    return residual + ab


class WideResnetShakeShakeGroup(nn.Module):
  """Defines a WideResnetGroup - Flax Linen version."""
  
  blocks_per_group: int
  channels: int
  strides: Tuple[int, int] = (1, 1)
  true_gradient: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
    """Forward pass."""
    for i in range(self.blocks_per_group):
      x = ShakeShakeBlock(
          channels=self.channels,
          strides=self.strides if i == 0 else (1, 1),
          true_gradient=self.true_gradient,
          name=f'block_{i}')(x, train=train)
    return x


class WideResnetShakeShake(nn.Module):
  """Defines the WideResnet Model with ShakeShake - Flax Linen version."""
  
  blocks_per_group: int
  channel_multiplier: int
  num_outputs: int
  true_gradient: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
    """Forward pass."""
    # Initial conv
    conv_init = nn.Conv(
        features=16,
        kernel_size=(3, 3),
        padding='SAME',
        kernel_init=utils.conv_kernel_init_fn,
        use_bias=False,
        name='init_conv')
    x = conv_init(x)
    
    bn_init = nn.BatchNorm(use_running_average=not train, name='init_bn')
    x = bn_init(x)
    
    # Groups
    x = WideResnetShakeShakeGroup(
        blocks_per_group=self.blocks_per_group,
        channels=16 * self.channel_multiplier,
        true_gradient=self.true_gradient,
        name='group_0')(x, train=train)
    
    x = WideResnetShakeShakeGroup(
        blocks_per_group=self.blocks_per_group,
        channels=32 * self.channel_multiplier,
        strides=(2, 2),
        true_gradient=self.true_gradient,
        name='group_1')(x, train=train)
    
    x = WideResnetShakeShakeGroup(
        blocks_per_group=self.blocks_per_group,
        channels=64 * self.channel_multiplier,
        strides=(2, 2),
        true_gradient=self.true_gradient,
        name='group_2')(x, train=train)
    
    # Final layers
    x = jax.nn.relu(x)
    x = nn.avg_pool(x, x.shape[1:3])
    x = x.reshape((x.shape[0], -1))
    
    dense = nn.Dense(
        features=self.num_outputs,
        kernel_init=utils.dense_layer_init_fn,
        name='dense')
    x = dense(x)
    return x

