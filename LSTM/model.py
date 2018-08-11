#
# File: model.py
# Date:21.01.2017
# Author: Denis Tananaev
# 
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers

#import os
#import re
#import tarfile
#import math 

import numpy as np
#layers
import sys
sys.path.insert(0, '../layers/')
import layers.summary as sm
import layers.conv as cnv
import layers.deconv as dcnv
import layers.batchnorm as bn
from layers.convLSTM import basic_conv_lstm_cell_leakyrelu_norm
import LSTM.param as param

#parameters of the datasets
IMAGE_SIZE_W=param.IMAGE_SIZE_W
IMAGE_SIZE_H=param.IMAGE_SIZE_H

#Training parameters
BATCH_SIZE=param.BATCH_SIZE
FLOAT16=param.FLOAT16
WEIGHT_DECAY=param.WEIGHT_DECAY
sequence_len=param.SEQUENCE_LEN


def inference(images, scope='RNN'):
        #train network    
    with tf.name_scope(scope, [images]):
        depth_scale1=[]
        depth_scale2=[]   
        depth_scale3=[]
        normals_scale1=[]
        normals_scale2=[]   
        #transform images to NHWC format
        images=tf.reshape(images,[BATCH_SIZE,sequence_len,3,IMAGE_SIZE_H,IMAGE_SIZE_W])
        images= tf.transpose(images,[0,1,3,4,2])
        
        #Initialize memory of the LSTM units      
        lstm_state1,lstm_state2,lstm_state3,lstm_state4,lstm_state5 = None, None, None, None,None
        lstm_state6,lstm_state7,lstm_state8, lstm_state9,lstm_state10,lstm_state11 = None, None,None,None,None,None

        for i in range(0,sequence_len):
            
            reuse = bool(depth_scale3) 
            
            im=images[:,i,:,:,:]    
            
            #======================#                                       
            #Scale 1: coarse level #
            #======================#  
        
            #======Output size: 96x128x64
            #Layer 1:
            conv1=cnv.conv(im,'conv1',[3, 3, 3, 64],stride=[1,2,2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16,reuse=reuse)
            relu1=tf.nn.leaky_relu(conv1,alpha=0.1)            
            relu1 = tf_layers.layer_norm(relu1, scope='layer_norm1',reuse=reuse)
        
            #Layer 2:
            hidden1, lstm_state1 = basic_conv_lstm_cell_leakyrelu_norm(relu1,lstm_state1,64,filter_size=5,scope="cLSTM1",reuse=reuse)
                        
            #======Output size: 48x64x128                             
            #Layer 3
            conv3=cnv.conv(hidden1,'conv3',[3, 3, 64, 128],stride=[1,2,2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16,reuse=reuse)
            relu3=tf.nn.leaky_relu(conv3,alpha=0.1)           
            relu3 = tf_layers.layer_norm(relu3, scope='layer_norm3',reuse=reuse) 
 
            #Layer 4
            hidden2, lstm_state2 = basic_conv_lstm_cell_leakyrelu_norm(relu3,lstm_state2,128,filter_size=5,scope="cLSTM2",reuse=reuse)
           
            #======Output size: 24x32x256     
            #Layer 5
            conv5=cnv.conv(hidden2,'conv5',[3, 3, 128, 256],stride=[1,2,2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16,reuse=reuse)
            relu5=tf.nn.leaky_relu(conv5,alpha=0.1)             
            relu5 = tf_layers.layer_norm(relu5, scope='layer_norm5',reuse=reuse)
            
            #Layer 6
            hidden3, lstm_state3 = basic_conv_lstm_cell_leakyrelu_norm(relu5,lstm_state3,256,filter_size=5,scope="cLSTM3",reuse=reuse)
            
            
    
            #======Output size: 12x16x512   
            #Layer 7
            conv7=cnv.conv(hidden3,'conv7',[3, 3, 256, 512],stride=[1,2,2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16,reuse=reuse)
            relu7=tf.nn.leaky_relu(conv7,alpha=0.1)            
            relu7 = tf_layers.layer_norm(relu7, scope='layer_norm7',reuse=reuse)
            
            #Layer 8
            hidden4, lstm_state4= basic_conv_lstm_cell_leakyrelu_norm(relu7,lstm_state4,512,filter_size=5,scope="cLSTM4",reuse=reuse)
        
            #======Output size: 6x8x512 
            #Layer 9
            conv9=cnv.conv(hidden4,'conv9',[3, 3, 512, 512],stride=[1,2,2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16,reuse=reuse)
            relu9=tf.nn.leaky_relu(conv9,alpha=0.1)            
            relu9 = tf_layers.layer_norm(relu9, scope='layer_norm9',reuse=reuse)
                
            #upasmpling
            #======Output size: 12x16x512 
            #Layer 10 
            conv10=dcnv.deconv(relu9,[BATCH_SIZE,int(IMAGE_SIZE_H/16),int(IMAGE_SIZE_W/16),512],'d_conv10',[4, 4, 512, 512],stride=[1, 2, 2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16,reuse=reuse)
            relu10=tf.nn.leaky_relu(conv10,alpha=0.1)
            relu10 = tf_layers.layer_norm(relu10, scope='layer_norm10',reuse=reuse)        
            #Layer 11        
            hidden5, lstm_state5= basic_conv_lstm_cell_leakyrelu_norm(relu10+hidden4,lstm_state5,512,filter_size=5,scope="cLSTM5",reuse=reuse)
            #======Output size: 24x32x256
            #Layer 12
            conv12=dcnv.deconv(hidden5,[BATCH_SIZE,int(IMAGE_SIZE_H/8),int(IMAGE_SIZE_W/8),256],'d_conv12',[4, 4, 256, 512],stride=[1, 2, 2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16,reuse=reuse)
            relu12=tf.nn.leaky_relu(conv12,alpha=0.1)  
            relu12 = tf_layers.layer_norm(relu12, scope='layer_norm12',reuse=reuse)   
            #Layer 13 
            hidden6, lstm_state6= basic_conv_lstm_cell_leakyrelu_norm(relu12+hidden3,lstm_state6,256,filter_size=5,scope="cLSTM6",reuse=reuse)             
            #======Output size: 48x64x256
            #Layer 14
            conv14=dcnv.deconv(hidden6,[BATCH_SIZE,int(IMAGE_SIZE_H/4),int(IMAGE_SIZE_W/4),128],'d_conv14',[4, 4, 128, 256],stride=[1, 2, 2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16,reuse=reuse)
            relu14=tf.nn.leaky_relu(conv14,alpha=0.1) 
            relu14 = tf_layers.layer_norm(relu14, scope='layer_norm14',reuse=reuse)   
            
            #Layer 15        
            conv15=cnv.conv(relu14+hidden2,'conv15',[3, 3, 128, 128],stride=[1,1,1, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16,reuse=reuse)
            relu15=tf.nn.leaky_relu(conv15,alpha=0.1)            
            relu15 = tf_layers.layer_norm(relu15, scope='layer_norm15',reuse=reuse)              
         
            #===================== output depth scale 1: 48x64x1 /4
            out_scale1=cnv.conv(relu15,'out_scale1',[3, 3, 128, 4],stride=[1,1,1, 1],padding='SAME',wd=0,FLOAT16=FLOAT16,reuse=reuse)    
            
         
            #======================#                                       
            #Scale 2: middle level #
            #======================#      
            
            #======Output size: 96x128x64
            #Layer 17:
            conv17=cnv.conv(im,'conv17',[3, 3, 3, 64],stride=[1,2,2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16,reuse=reuse)
            relu17=tf.nn.leaky_relu(conv17,alpha=0.1)           
            relu17 = tf_layers.layer_norm(relu17, scope='layer_norm17',reuse=reuse)            
            #Layer 18:        
            hidden7, lstm_state7= basic_conv_lstm_cell_leakyrelu_norm(relu17+relu1,lstm_state7,64,filter_size=5,scope="cLSTM7",reuse=reuse)
            
            #Layer 19:48x64x128
            conv19=cnv.conv(hidden7,'conv19',[3, 3, 64, 128],stride=[1,2,2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16,reuse=reuse)
            relu19=tf.nn.leaky_relu(conv19,alpha=0.1)        
            relu19 = tf_layers.layer_norm(relu19, scope='layer_norm19',reuse=reuse)   
            
            #Layer 20:        
            hidden8, lstm_state8= basic_conv_lstm_cell_leakyrelu_norm(relu19+relu3,lstm_state8,128,filter_size=5,scope="cLSTM8",reuse=reuse)             
            
            
            #concatenate featuremap from coarse level
            concat1= tf.concat([hidden8,relu15,out_scale1], 3, name='concat1')
        
            #Layer 21:        
            conv21=cnv.conv(concat1,'conv21',[9, 9, 260, 256],stride=[1,2,2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16,reuse=reuse)
            relu21=tf.nn.leaky_relu(conv21,alpha=0.1)           
            relu21 = tf_layers.layer_norm(relu21, scope='layer_norm21',reuse=reuse)   
            
            #upsampling
            #Layer 22
            conv22=dcnv.deconv(relu21,[BATCH_SIZE,int(IMAGE_SIZE_H/4),int(IMAGE_SIZE_W/4),128],'d_conv22',[4, 4, 128, 256],stride=[1, 2, 2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16,reuse=reuse)
            relu22=tf.nn.leaky_relu(conv22,alpha=0.1)   
            relu22 = tf_layers.layer_norm(relu22, scope='layer_norm22',reuse=reuse)                

            #Layer 23:        
            hidden9, lstm_state9= basic_conv_lstm_cell_leakyrelu_norm(relu22+hidden8,lstm_state9,128,filter_size=5,scope="cLSTM9",reuse=reuse)
            
            #Layer 24:   
            conv24=dcnv.deconv(hidden9,[BATCH_SIZE,int(IMAGE_SIZE_H/2),int(IMAGE_SIZE_W/2),64],'d_conv24',[4, 4, 64, 128],stride=[1, 2, 2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16,reuse=reuse)
            relu24=tf.nn.leaky_relu(conv24,alpha=0.1)    
            relu24 = tf_layers.layer_norm(relu24, scope='layer_norm24',reuse=reuse)             
            
            conv25=cnv.conv(relu24,'conv25',[5, 5, 64, 64],stride=[1,1,1, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16,reuse=reuse)
            relu25=tf.nn.leaky_relu(conv25,alpha=0.1)           
            relu25 = tf_layers.layer_norm(relu25, scope='layer_norm25',reuse=reuse) 
        
            #===================== output depth scale 2: 96x128x1 /4
            out_scale2=cnv.conv(relu25,'out_scale2',[3, 3, 64, 4],stride=[1,1,1, 1],padding='SAME',wd=0,FLOAT16=FLOAT16,reuse=reuse)        
        
        
            #======================#                                       
            #Scale 3: fine level   #
            #======================#         
            #======Output size:  192x256x32
            #Layer 27:
            conv27=cnv.conv(im,'conv27',[3, 3, 3, 32],stride=[1,1,1, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16,reuse=reuse)
            relu27=tf.nn.leaky_relu(conv27,alpha=0.1)        
            relu27 = tf_layers.layer_norm(relu27, scope='layer_norm27',reuse=reuse)    
            
            #Layer 28:     
            conv28=cnv.conv(relu27,'conv28',[3, 3, 32, 64],stride=[1,2,2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16,reuse=reuse)
            relu28=tf.nn.leaky_relu(conv28,alpha=0.1)         
            relu28 = tf_layers.layer_norm(relu28, scope='layer_norm28',reuse=reuse)            
        
            #Layer 29:        
            hidden10, lstm_state10= basic_conv_lstm_cell_leakyrelu_norm(relu28+relu17,lstm_state10,64,filter_size=5,scope="cLSTM10",reuse=reuse)      
        
            #concatenate featuremap from middle leveil
            concat2= tf.concat([hidden10,relu25,out_scale2], 3, name='concat2')    
        
            #Layer 30:      
            conv30=cnv.conv(concat2,'conv30',[5, 5, 132, 128],stride=[1,2,2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16,reuse=reuse)
            relu30=tf.nn.leaky_relu(conv30,alpha=0.1)            
            relu30 = tf_layers.layer_norm(relu30, scope='layer_norm30',reuse=reuse) 
            
            #Layer 31:   
            conv31=dcnv.deconv(relu30,[BATCH_SIZE,int(IMAGE_SIZE_H/2),int(IMAGE_SIZE_W/2),64],'d_conv31',[4, 4, 64, 128],stride=[1, 2, 2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16,reuse=reuse)
            relu31=tf.nn.leaky_relu(conv31,alpha=0.1)      
            relu31 = tf_layers.layer_norm(relu31, scope='layer_norm31',reuse=reuse)              
        
        
            #Layer 32:      
            hidden11, lstm_state11= basic_conv_lstm_cell_leakyrelu_norm(relu31+hidden10,lstm_state11,64,filter_size=5,scope="cLSTM11",reuse=reuse)    
            
            #Layer 33:   
            conv33=dcnv.deconv(hidden11,[BATCH_SIZE,int(IMAGE_SIZE_H),int(IMAGE_SIZE_W),32],'d_conv33',[4, 4, 32, 64],stride=[1, 2, 2, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16,reuse=reuse)
            relu33=tf.nn.leaky_relu(conv33,alpha=0.1)      
            relu33 = tf_layers.layer_norm(relu33, scope='layer_norm33',reuse=reuse)    
            
            #Layer 34:      
            conv34=cnv.conv(relu33+relu27,'conv34',[3, 3, 32, 32],stride=[1,1,1, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16,reuse=reuse)
            relu34=tf.nn.leaky_relu(conv34,alpha=0.1)      
            relu34 = tf_layers.layer_norm(relu34, scope='layer_norm34',reuse=reuse)         
            #Layer 35:      
            conv35=cnv.conv(relu34,'conv35',[3, 3, 32, 32],stride=[1,1,1, 1],padding='SAME',wd=WEIGHT_DECAY,FLOAT16=FLOAT16,reuse=reuse)
            relu35=tf.nn.leaky_relu(conv35,alpha=0.1)            
            relu35 = tf_layers.layer_norm(relu35, scope='layer_norm35',reuse=reuse)           
            #Inference layer 11 
            depth=cnv.conv(relu35,'depth',[3, 3, 32, 1],wd=0,FLOAT16=FLOAT16,reuse=reuse)
      
            #split
            scale1_depth=out_scale1[:,:,:,0]
            scale1_depth=tf.expand_dims(scale1_depth,3)        
            depth_scale1.append(scale1_depth)

        
            norm_x1=out_scale1[:,:,:,1]
            norm_y1=out_scale1[:,:,:,2]
            norm_z1=out_scale1[:,:,:,3]
            norm_x1=tf.expand_dims(norm_x1,3)
            norm_y1=tf.expand_dims(norm_y1,3)            
            norm_z1=tf.expand_dims(norm_z1,3)          
            scale1_normal=tf.concat([norm_x1,norm_y1,norm_z1], 3)
            normals_scale1.append(scale1_normal)
        
            tf.summary.image('depth_scale1:', scale1_depth)   
            tf.summary.image('normal_scale1:',scale1_normal)
        
        
        
            scale2_depth=out_scale2[:,:,:,0]
            scale2_depth=tf.expand_dims(scale2_depth,3)
            depth_scale2.append(scale2_depth)

            norm_x2=out_scale2[:,:,:,1]
            norm_y2=out_scale2[:,:,:,2]
            norm_z2=out_scale2[:,:,:,3]
            norm_x2=tf.expand_dims(norm_x2,3)
            norm_y2=tf.expand_dims(norm_y2,3)            
            norm_z2=tf.expand_dims(norm_z2,3)         
            scale2_normal=tf.concat([norm_x2,norm_y2,norm_z2], 3)
            normals_scale2.append(scale2_normal)      
            tf.summary.image('depth_scale2:', scale2_depth)   
            tf.summary.image('normal_scale2:',scale2_normal)   
        
        
            depth_scale3.append(depth)
            tf.summary.image('depth_scale3:', depth)   
        
        return depth_scale1, depth_scale2, depth_scale3,normals_scale1,normals_scale2

