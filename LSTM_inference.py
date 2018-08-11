#
# File: LSTM_inference.py
# Date:11.08.2018
# Author: Denis Tananaev
# 
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#Tensorflow
import tensorflow as tf
from tensorflow.python.framework import ops
#python lib
import numpy as np
import scipy.misc
import numpngw
#from PIL import Image
#import matplotlib.pyplot as plt
#from datetime import datetime
#import math
#import os.path
import time
import  LSTM.param as param
import  LSTM.model as model
import  LSTM.load_seq as load_seq
#System
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#parameters
BATCH_SIZE=param.BATCH_SIZE
#parameters
SaveDepthRealValued=False
CHECKPOINT_DIR="./LSTM_checkpoint/"
data_folder="./example/"
result_folder="./example/"

 

def inverse(depth):
    inverse=tf.divide(tf.ones_like(depth),depth)
    inverse=tf.where(tf.is_nan(inverse),tf.zeros_like(inverse),inverse)
    return inverse

def save_depth(result,config,saver,output_path):
  
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
            print("Loading model and evaluation took (seconds): ", duration)
            
            res=np.array(res)
            print("Minimum value of depth (meters): ",np.min(res))
            print("Maximum value of depth (meters): ",np.max(res))                        
            res*=1000
  
            for i in range(0,res.shape[4]):
                if i<10:
                    image_name="LSTM_depth_0"+str(i)+".png"
                else:
                    image_name="LSTM_depth_"+str(i)+".png"

                name=output_path+image_name
                
                depth=np.array(res[0,0,:,:,i],dtype=np.uint16)
                
                if SaveDepthRealValued==True:
                    numpngw.write_png(name,depth)
                else:
                    scipy.misc.toimage(depth, cmin=0, cmax=5000).save(name)

        

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
        print("Result depths saved in ./example/ folder")                      

    
def predict(input_path,output_path):
    
  #get input image    
  ops.reset_default_graph()
  with tf.Graph().as_default() as g:
    input_images=load_seq.batch_inputs(input_path)
    depth_scale1, depth_scale2, depth_scale3,normals_scale1,normals_scale2= model.inference(input_images)
    result = inverse(depth_scale3)
    result=tf.transpose(result,[1,2,3,0,4])
    result=result[:,:,:,:,0]

    # Restore the model for eval.
    saver = tf.train.Saver()

    #Allocate 70% of the GPU memory for inference (optional)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config = tf.ConfigProto(gpu_options=gpu_options)
    
    save_depth(result,config,saver,output_path)

def main(argv=None): 
  predict(data_folder,result_folder)


if __name__ == '__main__':
  tf.app.run() 
