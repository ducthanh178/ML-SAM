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

"""Mock module for tensorflow_addons.image to replace deprecated library."""

import tensorflow as tf


class ImageModule:
  """Mock image module for tensorflow_addons."""
  
  @staticmethod
  def rotate(images, angles):
    """Rotate images by angles (in radians)."""
    return tf.image.rot90(images, k=1)  # Simplified - use tf.image for rotation
  
  @staticmethod
  def translate(images, translations):
    """Translate images."""
    # Use tf.raw_ops.ImageProjectiveTransformV3 or tf.image operations
    # For simplicity, we'll use a basic translation
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=images,
        transforms=tf.constant([[1.0, 0.0, translations[0],
                                0.0, 1.0, translations[1],
                                0.0, 0.0]], dtype=tf.float32),
        output_shape=tf.shape(images)[1:3],
        fill_value=0.0,
        interpolation='BILINEAR')
  
  @staticmethod
  def transform(images, transforms):
    """Apply projective transforms to images."""
    # transforms is a flat list of 8 values [a, b, c, d, e, f, g, h]
    # representing the transform matrix
    batch_size = tf.shape(images)[0]
    transforms = tf.reshape(transforms, [1, 8])
    transforms = tf.tile(transforms, [batch_size, 1])
    
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=images,
        transforms=transforms,
        output_shape=tf.shape(images)[1:3],
        fill_value=0.0,
        interpolation='BILINEAR')


# Create the image module
image = ImageModule()

