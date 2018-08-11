#
# Author: Denis Tananaev
# File: activations.py
# Date: 9.02.2017
# Description: activation functions for neural networks
#

#include libs
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#import os
from six.moves import xrange
import tensorflow as tf
#import summary
import layers.summary as sm

def ReLU(x,scope, reuse=None):
    with tf.variable_scope(scope, 'ReLU', [x], reuse=reuse):
        relu= tf.nn.relu(x)
        sm._activation_summary(relu)
        return relu  

def leakyReLU(x,leak, scope, reuse=None):
    with tf.variable_scope(scope, 'leakyReLU', [x], reuse=reuse):
        leakyrelu= tf.nn.leaky_relu(x,alpha=leak,name=None)
        sm._activation_summary(leakyrelu)
        return leakyrelu  
