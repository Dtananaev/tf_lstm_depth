
import os
import sys
import numpy as np
from PIL import Image
import glob
import predict
data_folder="/misc/lmbraid11/tananaed/Article/1.Evaluation/evaluation_50_NYU/"
result_folder="./evaluation50/"


def read_file(textfile):
  '''Read txt file and output array of strings line by line '''
  with open(textfile) as f:
    result = f.read().splitlines()
  return result

def get_folders(data_folder):
    res=[]
    folder_list=glob.glob(data_folder+"*")
    for l in folder_list:
        name=os.path.basename(l)
        res.append(name)
    return res
#read folders
folders=get_folders(data_folder)
def get_index(n):
    string=[]
    i=n+1
    if (i<10):
        string='0'*6
    elif(i<100):
        string='0'*5  
    elif (i<1000):
        string='0'*4  
    elif (i<10000): 
        string='0'*3   
    return string

for folder in folders:
    print(folder)
    tsc_inv=0
    tL1_rel=0
    tL1_inv=0
    for i in range(50):
        string=get_index(i)
        input_path=data_folder+folder+"/"
        output_path=result_folder+folder+"/"
        image_name="i_"+string+str(i+1)+".png"
        
        if not os.path.exists(result_folder+folder):
            os.makedirs(result_folder+folder)
        predict.predict(input_path,output_path,image_name)
