#
# File: load_image.py
# Date:21.01.2017
# Author: Denis Tananaev
# 
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import CNN.param  as param
import numpy as np
#import glob


#parameter
IMAGE_SIZE_W=param.IMAGE_SIZE_W
IMAGE_SIZE_H=param.IMAGE_SIZE_H
FLOAT16=param.FLOAT16
#parameters of the data uploading
BATCH_SIZE=param.BATCH_SIZE


def read_images_from_disk(input_image):
  image = tf.read_file(input_image)
  example_image = tf.image.decode_png(image, channels=3)
  example_image.set_shape([480,640 , 3])
  
  example_image = tf.cast(example_image, tf.int32)
  example_image=tf.image.crop_to_bounding_box(example_image,44, 40, 427,561)
  channels = tf.unstack (example_image, axis=-1)
  image= tf.stack([channels[2], channels[1], channels[0]], axis=-1)
  return image

def input_image(image_name):

    with tf.name_scope('image_processing'):
      image = read_images_from_disk(image_name)
      image=tf.expand_dims(image,0)
      image=tf.image.resize_images(image,[IMAGE_SIZE_H,IMAGE_SIZE_W])
      image= tf.transpose(image,[0,3,1,2])
      image=image/255 -0.5
      return image