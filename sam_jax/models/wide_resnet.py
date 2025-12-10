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

"""Wide Resnet Model - Migrated to Flax Linen API.

Reference:
Wide Residual Networks, Sergey Zagoruyko, Nikos Komodakis
https://arxiv.org/abs/1605.07146
"""

from typing import Tuple
from flax import linen as nn
import jax.numpy as jnp
from sam_new.sam_jax.models import utils


def _output_add(block_x: jnp.ndarray, orig_x: jnp.ndarray) -> jnp.ndarray:
  """Add two tensors, padding them with zeros or pooling them if necessary."""
  stride = orig_x.shape[-2] // block_x.shape[-2]
  strides = (stride, stride)
  if block_x.shape[-1] != orig_x.shape[-1]:
    orig_x = nn.avg_pool(orig_x, strides, strides)
    channels_to_add = block_x.shape[-1] - orig_x.shape[-1]
    orig_x = jnp.pad(orig_x, [(0, 0), (0, 0), (0, 0), (0, channels_to_add)])
  return block_x + orig_x


class WideResnetBlock(nn.Module):
  """Defines a single WideResnetBlock - Flax Linen version."""
  
  channels: int
  strides: Tuple[int, int] = (1, 1)
  activate_before_residual: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
    """Forward pass."""
    if self.activate_before_residual:
      bn = nn.BatchNorm(use_running_average=not train, name='init_bn')
      x = bn(x)
      x = nn.relu(x)
      orig_x = x
    else:
      orig_x = x

    block_x = x
    if not self.activate_before_residual:
      bn1 = nn.BatchNorm(use_running_average=not train, name='init_bn')
      block_x = bn1(block_x)
      block_x = nn.relu(block_x)

    conv1 = nn.Conv(
        features=self.channels,
        kernel_size=(3, 3),
        strides=self.strides,
        padding='SAME',
        use_bias=False,
        kernel_init=utils.conv_kernel_init_fn,
        name='conv1')
    block_x = conv1(block_x)
    
    bn2 = nn.BatchNorm(use_running_average=not train, name='bn_2')
    block_x = bn2(block_x)
    block_x = nn.relu(block_x)
    
    conv2 = nn.Conv(
        features=self.channels,
        kernel_size=(3, 3),
        padding='SAME',
        use_bias=False,
        kernel_init=utils.conv_kernel_init_fn,
        name='conv2')
    block_x = conv2(block_x)

    return _output_add(block_x, orig_x)


class WideResnetGroup(nn.Module):
  """Defines a WideResnetGroup - Flax Linen version."""
  
  blocks_per_group: int
  channels: int
  strides: Tuple[int, int] = (1, 1)
  activate_before_residual: bool = False
  use_additional_skip_connections: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
    """Forward pass."""
    orig_x = x
    for i in range(self.blocks_per_group):
      x = WideResnetBlock(
          channels=self.channels,
          strides=self.strides if i == 0 else (1, 1),
          activate_before_residual=self.activate_before_residual and (i == 0),
          name=f'block_{i}')(x, train=train)
    if self.use_additional_skip_connections:
      x = _output_add(x, orig_x)
    return x


class WideResnet(nn.Module):
  """Defines the WideResnet Model - Flax Linen version."""
  
  blocks_per_group: int
  channel_multiplier: int
  num_outputs: int
  use_additional_skip_connections: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
    """Forward pass."""
    first_x = x
    
    # Initial conv
    conv_init = nn.Conv(
        features=16,
        kernel_size=(3, 3),
        padding='SAME',
        name='init_conv',
        kernel_init=utils.conv_kernel_init_fn,
        use_bias=False)
    x = conv_init(x)
    
    # Groups
    x = WideResnetGroup(
        blocks_per_group=self.blocks_per_group,
        channels=16 * self.channel_multiplier,
        activate_before_residual=True,
        use_additional_skip_connections=self.use_additional_skip_connections,
        name='group_0')(x, train=train)
    
    x = WideResnetGroup(
        blocks_per_group=self.blocks_per_group,
        channels=32 * self.channel_multiplier,
        strides=(2, 2),
        use_additional_skip_connections=self.use_additional_skip_connections,
        name='group_1')(x, train=train)
    
    x = WideResnetGroup(
        blocks_per_group=self.blocks_per_group,
        channels=64 * self.channel_multiplier,
        strides=(2, 2),
        use_additional_skip_connections=self.use_additional_skip_connections,
        name='group_2')(x, train=train)
    
    if self.use_additional_skip_connections:
      x = _output_add(x, first_x)
    
    # Final layers
    bn_final = nn.BatchNorm(use_running_average=not train, name='pre-pool-bn')
    x = bn_final(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, x.shape[1:3])
    x = x.reshape((x.shape[0], -1))
    
    dense = nn.Dense(
        features=self.num_outputs,
        kernel_init=utils.dense_layer_init_fn,
        name='dense')
    x = dense(x)
    return x

