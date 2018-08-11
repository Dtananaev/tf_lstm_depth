#
# File: load_seq.py
# Date:21.01.2017
# Author: Denis Tananaev
# 
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np
import glob
import os

#import parameters file
import LSTM.param as param

#parameters of the datasets
IMAGE_SIZE_W=param.IMAGE_SIZE_W
IMAGE_SIZE_H=param.IMAGE_SIZE_H
FLOAT16=param.FLOAT16
#parameters of the data uploading
BATCH_SIZE=param.BATCH_SIZE
NUM_READERS=1
sequence=param.SEQUENCE_LEN


#read NYU image 
def read_images_from_disk(input_image):
  #read image
  image = tf.read_file(input_image)
  #decode png
  example_image = tf.image.decode_png(image, channels=3)
  #reshape to the default NYU resolution 640x480
  example_image.set_shape([480,640 , 3])
  #type cast
  example_image = tf.cast(example_image, tf.int32)
  #crop image borders by using NYU mask
  example_image= tf.image.crop_to_bounding_box(example_image,44, 40, 427,561)
  #convert from RGB to BGR
  channels = tf.unstack (example_image, axis=-1)
  image= tf.stack([channels[2], channels[1], channels[0]], axis=-1)
  return image


def get_images(path):
    images=glob.glob(path+"i_*.png")
    return images

def batch_inputs(path):

    with tf.name_scope('batch_processing'):
        l_im=get_images(path)  
        l_im=sorted(l_im)
        for i in range(param.SEQUENCE_LEN):
	    print("image ",l_im[i])
        #read the list of pathes
        for i in range(sequence):
            image = read_images_from_disk(l_im[i])
            if i==0:
                images=image
            else:
                images= tf.concat([images,image], 2, name='concat1')#concatenate sequence for LSTM   
                
        images=tf.expand_dims(images,0)
        images=tf.image.resize_images(images,[IMAGE_SIZE_H,IMAGE_SIZE_W])
        images= tf.transpose(images,[0,3,1,2])
        images=images/255 -0.5
    return images








