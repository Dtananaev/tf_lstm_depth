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
We trained CNN and LSTM networks of the same architecture for single frame depth prediction. The architecture of the LSTM network are shown below.
<p align="center">
  <img src="https://github.com/Dtananaev/tf_lstm_depth/blob/master/pictures/Architecture.jpg" />
</p>
## Citation

If you use this method in your research, please cite:

    @inproceedings{Tananaev2018tempo,
            title={Temporally consistent depth estimation in videos with recurrent architectures},
            author={Denis Tananaev, Huizhong Zhou, Benjamin Ummenhofer,Thomas Brox},
            booktitle={3D Reconstruction meets Semantics, ECCV 2018 Workshop},
            year={2018}
    }
