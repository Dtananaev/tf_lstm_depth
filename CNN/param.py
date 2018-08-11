#
# File: param.py
# Date:21.01.2017
# Author: Denis Tananaev
# 
#

#the parameters of dataset
IMAGE_SIZE_W=256
IMAGE_SIZE_H=192
FLOAT16=False

#the parameters data upload
BATCH_SIZE=1
SEQUENCE_LEN=50

#Training algorithm (Adam or SGRD)
ADAM=True
STARTER_LEARNING_RATE=0.0001
#Training property (SGD with momentum and restarts)
MOMENTUM=0.9
INITIAL_PERIOD_STEPS=30000 #iterations
T_MUL=2
M_MUL=0.8

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.0  # The decay to use for the moving average.
WEIGHT_DECAY=0.00001
#training and test
TRAIN_LOG="./log"
TEST_LOG="./eval"
BEST_CHECKPOINT="./log/best_checkpoint"

#additional
LOG_DEVICE_PLACEMENT=False

#Number of steps before validation
STEPS_VALID=2000

#Split test and validation dataset
validation_set=[0,7,31,34,35,36,39,49,91,96,97,195]
