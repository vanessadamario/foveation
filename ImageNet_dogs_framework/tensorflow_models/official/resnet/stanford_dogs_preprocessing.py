# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities to preprocess images.

Training images are sampled using the provided bounding boxes, and subsequently
cropped to the sampled bounding box. Images are additionally flipped randomly,
then resized to the target output size (without aspect-ratio preservation).

Images used during evaluation are resized (with aspect-ratio preservation) and
centrally cropped.

All images undergo mean color subtraction.

Note that these steps are colloquially referred to as "ResNet preprocessing,"
and they differ from "VGG preprocessing," which does not use bounding boxes
and instead does an aspect-preserving resize followed by random crop during
training. (These both differ from "Inception preprocessing," which introduces
color distortion steps.)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

# The lower bound for the smallest side of the image for aspect-preserving
# resizing. For example, if an image is 500 x 1000, it will be resized to
# _RESIZE_MIN x (_RESIZE_MIN * 2).
_RESIZE_MIN = 256


def _crop_bounding_box(image, bbox):
  """
  Function to generate cropped 3D image. We remove what is not in the bounding
  box, and pad the cropped image, so to preserve the original dimension before the cropping.
  Parameters
      :image: a 3D image tf.tensor
      :bbox: bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
    where each coordinate is [0, 1) and the coordinates are arranged as
    [ymin, xmin, ymax, xmax]
  Return
      cropped_padded_image: a 3D tf.tensor of the same dimension as the original image
  """
  ymin = bbox[0, 0, 0]
  xmin = bbox[0, 0, 1]
  ymax = bbox[0, 0, 2]
  xmax = bbox[0, 0, 3]

  shape = tf.shape(image)
  im_height, im_width = tf.cast(shape[0], tf.float32), tf.cast(shape[1], tf.float32)
  (xminn, xmaxx, yminn, ymaxx) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
  yminn = tf.cast(yminn, tf.int32)
  ymaxx = tf.cast(ymaxx, tf.int32)
  xminn = tf.cast(xminn, tf.int32)
  xmaxx = tf.cast(xmaxx, tf.int32)
  cropped = tf.image.crop_to_bounding_box(image,
                                          yminn,
                                          xminn,
                                          ymaxx - yminn,
                                          xmaxx - xminn)
  padding_chs = tf.fill([2], 0)  # a Tensor with values [0, 0]
  paddings = ([[yminn, shape[0] - ymaxx],
               [xminn, shape[1] - xmaxx],
               padding_chs])
  padded = tf.pad(cropped, paddings, mode='CONSTANT', name=None, constant_values=0)
  return padded


def _resize_image(image, height, width):
  """Simple wrapper around tf.resize_images.

  This is primarily to make sure we use the same `ResizeMethod` and other
  details each time.

  Args:
    image: A 3-D image `Tensor`.
    height: The target height for the resized image.
    width: The target width for the resized image.

  Returns:
    resized_image: A 3-D tensor containing the resized image. The first two
      dimensions have the shape [height, width].
  """
  return tf.image.resize_images(
      image, [height, width], method=tf.image.ResizeMethod.BILINEAR,
      align_corners=False)


def _mean_image_subtraction(image, means, num_channels):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.
    num_channels: number of color channels in the image that will be distorted.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')

  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  # We have a 1-D tensor of means; convert to 3-D.
  means = tf.expand_dims(tf.expand_dims(means, 0), 0)

  return image - means


def preprocess_image(image_buffer, bbox, output_height, output_width,
                     num_channels, is_training, crop=False):
  """Pre processes the given image.

  Pre processing includes decoding, cropping, and resizing for both training
  and eval images.
  Training pre processing, however, introduces some random distortion of the image to improve accuracy.

  Args:
    image_buffer: scalar string Tensor representing the raw JPEG image buffer.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    num_channels: Integer depth of the image buffer for decoding.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.
    crop: bool flag to crop the image or not
  Returns:
    A preprocessed image.
  """
  print("")
  print("")
  print("")
  print("")
  print("CROP")
  print(crop)
  image = tf.io.decode_jpeg(image_buffer, channels=num_channels)
  if crop:
    image = _crop_bounding_box(image, bbox)

  image = _resize_image(image, output_height, output_width)
  image.set_shape([output_height, output_width, num_channels])

  return _mean_image_subtraction(image, _CHANNEL_MEANS, num_channels)