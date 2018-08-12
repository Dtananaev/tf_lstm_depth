# Temporally Consistent Depth Estimation in Videos with Recurrent Architectures


Authors: [Denis Tananaev](http://denis.tananaev.eu/), [Huizhong Zhou](https://lmb.informatik.uni-freiburg.de/people/zhouh/), [Benjamin Ummenhofer](https://lmb.informatik.uni-freiburg.de/people/ummenhof/), [Thomas Brox](https://lmb.informatik.uni-freiburg.de/people/brox/).

[![Build Status](https://travis-ci.org/Dtananaev/localization.svg?branch=master)](https://travis-ci.org/Dtananaev/localization)
[![BSD2 License](http://img.shields.io/badge/license-BSD2-brightgreen.svg)](https://github.com/Dtananaev/localization/blob/master/LICENSE.md) 

## Content
1. [Introduction](#introduction)<br />
2. [Models](#models)<br />
3. [Guide](#quick-guide)<br />
4. [Results](#results)<br />
5. [Citation](#citation)


## Introduction

This repository contains the CNN and LSTM models trained for depth prediction from a single RGB image, as described in the paper "[Temporally Consistent Depth Estimation in Videos with Recurrent Architectures]()". The provided models are those that were used to obtain the results reported in the paper on the benchmark dataset NYU Depth v2.

To see overview  video of the work click on the picture below:
 [![introvideo](https://github.com/Dtananaev/tf_lstm_depth/blob/master/pictures/sfm.jpg)](https://youtu.be/r6k4JaV41xg)

## Models
We trained CNN and LSTM networks of the same architecture for single frame depth prediction. The architecture of the LSTM network is shown below.
<p align="center">
  <img src="https://github.com/Dtananaev/tf_lstm_depth/blob/master/pictures/Architecture.jpg" />
</p>

You can download weights for tensorflow for [CNN](https://drive.google.com/open?id=1Ib_2Lspy_p_8zhbSauMeIyKbKSksqpGd) (132 mb) and for RNN (601 mb) trained on NYUv2 dataset.

## Guide

The code was checked on Ubuntu 16.04 LTS with CUDA 9.0, CuDNN 7.05.15 and tensorflow 1.9 for python 2.7.
 * In order to run LSTM network unpack the LSTM tensorflow weight to the LSTM_checkpoint folder and run LSTM_inference.py the result depths will be saved in example folder.
 * In order to run CNN network unpack the CNN tensorflow weight to the CNN_checkpoint folder and run CNN_inference.py the result depths will be saved in example folder.
 
 To make 3D reconstruction we used [structure from motion (SfM)](https://github.com/PrincetonVision/SUN3Dsfm) pipeline from [Sun3D dataset](http://sun3d.cs.princeton.edu/). In order to apply SfM set  extrinsics parameters for Kinect sensor (can be found in [NYUv2 dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) scripts) normalized with respect to the output resolution.
 
## Results

In the following table, we report the results that should be obtained after evaluation on the uniformly sampled from NYUv2 test set [list](https://github.com/Dtananaev/tf_lstm_depth/blob/master/pictures/list.txt) of sequences with the first 50 images from each sequence and also compare to other  methods on depth prediction from a single image. 
- Error metrics on NYU Depth v2:

| State of the art on NYU     |  L1-inv  |  log10  | RMS | delta<1.25 | delta<1.25^2 | delta<1.25^3| 
|-----------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| [Liu](https://arxiv.org/pdf/1511.06070.pdf)       |0.155 | 0.133 | 0.995| 56.1 % | 81.1% | 88.7%|
| [Eigen & Fergus](http://cs.nyu.edu/~deigen/dnl/)  | 0.130 | 0.108 |  0.885 | 69.0% | 86.9% | 91.6%|
| [Chakrabarti](https://projects.ayanc.org/mdepth/) | 0.116 | 0.095 | **0.808**| 77.1% |88.3% | 91.2% | 
| [Laina](https://arxiv.org/pdf/1606.00373.pdf) | 0.114 | 0.093 | 0.823| 77.9% |88.4% | **91.7%** | 
| Ours single-frame                        | 0.119 | 0.101 | 0.878 | 75.4% | 87.7% | 91.2%|
| **Ours LSTM**                       | **0.111** | **0.092** | 0.824 |**79.6%** | **88.9%** | 91.4%|

- The point clouds for sequences of 25 images:
<p align="center">
  <img src="https://github.com/Dtananaev/tf_lstm_depth/blob/master/pictures/3D.jpg" />
</p>

- The comparison of the depth maps:
<p align="center">
  <img src="https://github.com/Dtananaev/tf_lstm_depth/blob/master/pictures/depthresult.jpg" />
</p>



## Citation

If you use this method in your research, please cite:

    @inproceedings{Tananaev2018tempo,
            title={Temporally consistent depth estimation in videos with recurrent architectures},
            author={Denis Tananaev, Huizhong Zhou, Benjamin Ummenhofer,Thomas Brox},
            booktitle={3D Reconstruction meets Semantics, ECCV 2018 Workshop},
            year={2018}
    }
