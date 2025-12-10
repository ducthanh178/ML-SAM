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

"""Flax Linen implementation of ResNet V1."""

from flax import linen as nn
import jax.numpy as jnp


class ResNetBlock(nn.Module):
  """ResNet block - Flax Linen version."""
  
  filters: int
  strides: tuple = (1, 1)

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
    """Forward pass."""
    residual = x
    
    conv1 = nn.Conv(
        features=self.filters,
        kernel_size=(3, 3),
        strides=self.strides,
        padding='SAME',
        use_bias=False,
        name='conv1')
    y = conv1(x)
    
    norm1 = nn.BatchNorm(use_running_average=not train, name='norm1')
    y = norm1(y)
    y = nn.relu(y)
    
    conv2 = nn.Conv(
        features=self.filters,
        kernel_size=(3, 3),
        padding='SAME',
        use_bias=False,
        name='conv2')
    y = conv2(y)
    
    norm2 = nn.BatchNorm(
        use_running_average=not train,
        scale_init=nn.initializers.zeros,
        name='norm2')
    y = norm2(y)
    
    if residual.shape != y.shape:
      conv_proj = nn.Conv(
          features=self.filters,
          kernel_size=(1, 1),
          strides=self.strides,
          padding='SAME',
          use_bias=False,
          name='conv_proj')
      residual = conv_proj(residual)
      
      norm_proj = nn.BatchNorm(use_running_average=not train, name='norm_proj')
      residual = norm_proj(residual)
    
    return nn.relu(residual + y)


class BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block - Flax Linen version."""
  
  filters: int
  strides: tuple = (1, 1)

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
    """Forward pass."""
    residual = x
    
    # 1x1 conv
    conv1 = nn.Conv(
        features=self.filters,
        kernel_size=(1, 1),
        padding='SAME',
        use_bias=False,
        name='conv1')
    y = conv1(x)
    
    norm1 = nn.BatchNorm(use_running_average=not train, name='norm1')
    y = norm1(y)
    y = nn.relu(y)
    
    # 3x3 conv
    conv2 = nn.Conv(
        features=self.filters,
        kernel_size=(3, 3),
        strides=self.strides,
        padding='SAME',
        use_bias=False,
        name='conv2')
    y = conv2(y)
    
    norm2 = nn.BatchNorm(use_running_average=not train, name='norm2')
    y = norm2(y)
    y = nn.relu(y)
    
    # 1x1 conv expand
    conv3 = nn.Conv(
        features=self.filters * 4,
        kernel_size=(1, 1),
        padding='SAME',
        use_bias=False,
        name='conv3')
    y = conv3(y)
    
    norm3 = nn.BatchNorm(
        use_running_average=not train,
        scale_init=nn.initializers.zeros,
        name='norm3')
    y = norm3(y)
    
    if residual.shape != y.shape:
      conv_proj = nn.Conv(
          features=self.filters * 4,
          kernel_size=(1, 1),
          strides=self.strides,
          padding='SAME',
          use_bias=False,
          name='conv_proj')
      residual = conv_proj(residual)
      
      norm_proj = nn.BatchNorm(use_running_average=not train, name='norm_proj')
      residual = norm_proj(residual)
    
    return nn.relu(residual + y)


class ResNet(nn.Module):
  """ResNetV1 - Flax Linen version."""
  
  stage_sizes: tuple
  num_classes: int
  num_filters: int = 64
  block_cls: type = ResNetBlock

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
    """Forward pass."""
    # Initial conv
    conv_init = nn.Conv(
        features=self.num_filters,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding=[(3, 3), (3, 3)],
        use_bias=False,
        name='conv_init')
    x = conv_init(x)
    
    bn_init = nn.BatchNorm(use_running_average=not train, name='bn_init')
    x = bn_init(x)
    x = nn.relu(x)
    
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    
    # Stages
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(
            filters=self.num_filters * 2 ** i,
            strides=strides,
            name=f'stage_{i}_block_{j}')(x, train=train)
    
    # Final layers
    x = jnp.mean(x, axis=(1, 2))
    
    dense = nn.Dense(features=self.num_classes, name='dense')
    x = dense(x)
    return x


# Predefined ResNet variants
def ResNet18(num_classes: int = 1000, **kwargs):
  return ResNet(stage_sizes=[2, 2, 2, 2], num_classes=num_classes, block_cls=ResNetBlock, **kwargs)

def ResNet34(num_classes: int = 1000, **kwargs):
  return ResNet(stage_sizes=[3, 4, 6, 3], num_classes=num_classes, block_cls=ResNetBlock, **kwargs)

def ResNet50(num_classes: int = 1000, **kwargs):
  return ResNet(stage_sizes=[3, 4, 6, 3], num_classes=num_classes, block_cls=BottleneckResNetBlock, **kwargs)

def ResNet101(num_classes: int = 1000, **kwargs):
  return ResNet(stage_sizes=[3, 4, 23, 3], num_classes=num_classes, block_cls=BottleneckResNetBlock, **kwargs)

def ResNet152(num_classes: int = 1000, **kwargs):
  return ResNet(stage_sizes=[3, 8, 36, 3], num_classes=num_classes, block_cls=BottleneckResNetBlock, **kwargs)

