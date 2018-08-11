#
# File: CNN_inferece.py
# Date:11.08.2018
# Author: Denis Tananaev
# 
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import ops

import scipy.misc
from numpy import newaxis
from PIL import Image

#import math
#import os.path
import time
import numpy as np
import CNN.model as model
import glob
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import CNN.load_image as load_image
import numpngw

#parameters
SaveDepthRealValued=False
CHECKPOINT_DIR="./CNN_checkpoint/"
data_folder="./example/"
result_folder="./example/"

def inverse(depth):
    inverse=tf.divide(tf.ones_like(depth),depth)
    inverse=tf.where(tf.is_nan(inverse),tf.zeros_like(inverse),inverse)
    return inverse

def save_depth(result,config,saver,output_path,counter):
  
    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
             #Assuming model_checkpoint_path looks something like:
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('Checkpoint is loaded') 
        else:
            print('No checkpoint file found')
            return
        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

            start_time = time.time()
            res=sess.run([result])
            duration = time.time() - start_time
            print("Loading model and evaluation took (seconds): ",duration)

            res=np.array(res)
            print("Minimum value of depth (meters): ",np.min(res))
            print("Maximum value of depth (meters): ",np.max(res))                     
            res*=1000
           
            if counter<10:
                image_name="CNN_depth_0"+str(counter)+".png"
            else:
                image_name="CNN_depth_"+str(counter)+".png"

            name=output_path+image_name
            depth=np.array(res[0,0,:,:,0],dtype=np.uint16)
            if SaveDepthRealValued==True:
                numpngw.write_png(name,depth)
            else:
                scipy.misc.toimage(depth, cmin=0, cmax=5000).save(name)


        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
    
def predict(input_path,output_path,image_name,counter):
    
  #get input image    
  ops.reset_default_graph()
  with tf.Graph().as_default() as g:
    input_image=load_image.input_image(image_name)
    scale1_depth, scale2_depth, depth,scale1_normal,scale2_normal = model.inference(input_image)
    result = inverse(depth)


    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config = tf.ConfigProto(gpu_options=gpu_options)
    
    save_depth(result,config,saver,output_path,counter)

    

def main(argv=None):
  images=glob.glob(data_folder+"i_*.png")
  images=sorted(images)
  for i in range(len(images)):
    predict(data_folder,result_folder,images[i],i)


if __name__ == '__main__':
  tf.app.run() 
