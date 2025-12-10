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

"""Build FLAX models for image classification - Flax Linen version."""

from typing import Optional, Tuple, Dict, Any
from flax import linen as nn
from flax.core import freeze, unfreeze
import jax
from jax import numpy as jnp
from jax import random

from sam_new.sam_jax.models import wide_resnet
from sam_new.sam_jax.models import wide_resnet_shakeshake
from sam_new.sam_jax.models import pyramidnet

_AVAILABLE_MODEL_NAMES = [
    'WideResnet28x10',
    'WideResnet28x6_ShakeShake',
    'Pyramid_ShakeDrop',
    'WideResnet_mini',  # For testing/debugging purposes.
    'WideResnet_ShakeShake_mini',
    'Pyramid_ShakeDrop_mini',
]


def create_image_model(
    prng_key: jnp.ndarray,
    batch_size: int,
    image_size: int,
    module: nn.Module,
    num_channels: int = 3
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """Instantiates a FLAX model and its state - Flax Linen version.

  Args:
    prng_key: PRNG key to use to sample the initial weights.
    batch_size: Batch size that the model should expect.
    image_size: Dimension of the image (assumed to be squared).
    module: FLAX linen module describing the model to instantiate.
    num_channels: Number of channels for the images.

  Returns:
    A tuple of (params, batch_stats) where params are the model parameters
    and batch_stats are the batch normalization statistics.
  """
  input_shape = (batch_size, image_size, image_size, num_channels)
  dummy_input = jnp.zeros(input_shape, dtype=jnp.float32)
  
  # Initialize the model
  variables = module.init(prng_key, dummy_input, train=False)
  params = variables['params']
  batch_stats = variables.get('batch_stats', {})
  
  return params, batch_stats


def get_model(
    model_name: str,
    batch_size: int,
    image_size: int,
    num_classes: int,
    num_channels: int = 3,
    prng_key: Optional[jnp.ndarray] = None,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
  """Returns an initialized model of the chosen architecture.

  Args:
    model_name: Name of the architecture to use. Should be one of
      _AVAILABLE_MODEL_NAMES.
    batch_size: The batch size that the model should expect.
    image_size: Dimension of the image (assumed to be squared).
    num_classes: Dimension of the output layer.
    num_channels: Number of channels for the images.
    prng_key: PRNG key to use to sample the weights.

  Returns:
    A tuple of (model, params, batch_stats) where model is the Flax linen module,
    params are the model parameters, and batch_stats are the batch normalization
    statistics.

  Raises:
    ValueError if the name of the architecture is not recognized.
  """
  if model_name == 'WideResnet28x10':
    model = wide_resnet.WideResnet(
        blocks_per_group=4,
        channel_multiplier=10,
        num_outputs=num_classes)
  elif model_name == 'WideResnet28x6_ShakeShake':
    model = wide_resnet_shakeshake.WideResnetShakeShake(
        blocks_per_group=4,
        channel_multiplier=6,
        num_outputs=num_classes)
  elif model_name == 'Pyramid_ShakeDrop':
    model = pyramidnet.PyramidNetShakeDrop(num_outputs=num_classes)
  elif model_name == 'WideResnet_mini':  # For testing.
    model = wide_resnet.WideResnet(
        blocks_per_group=2,
        channel_multiplier=1,
        num_outputs=num_classes)
  elif model_name == 'WideResnet_ShakeShake_mini':  # For testing.
    model = wide_resnet_shakeshake.WideResnetShakeShake(
        blocks_per_group=2,
        channel_multiplier=1,
        num_outputs=num_classes)
  elif model_name == 'Pyramid_ShakeDrop_mini':
    model = pyramidnet.PyramidNetShakeDrop(
        num_outputs=num_classes,
        pyramid_depth=11)
  else:
    raise ValueError(f'Unrecognized model name: {model_name}')
  
  if prng_key is None:
    prng_key = random.PRNGKey(0)

  params, batch_stats = create_image_model(
      prng_key, batch_size, image_size, model, num_channels)
  
  return model, params, batch_stats

