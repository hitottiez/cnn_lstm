# Overview
This code is for action recognition using the network which consists of CNN and LSTM.
I implemented this code for practice because I have not written any deep learning code from scratch!

# Setup
## Environment
- Linux (I used Ubuntu 14.04)
- GPU (I used GTX TitanX x2)
- Keras + TensorFlow (I recommend to install Keras via docker)

## Dataset
UCF-101 dataset  
http://crcv.ucf.edu/data/UCF101.php

# Run
## Make train/val split file
https://github.com/yjxiong/temporal-segment-networks

## Convert video to RGB images or optical flow
https://github.com/yjxiong/temporal-segment-networks

## Train model
You can train the network.
```Shell
bash train.sh
```

# TODO
- fix the GPU memory allocation error

# Reference
## CNN + LSTM
- https://github.com/keras-team/keras/issues/401
- https://github.com/keras-team/keras/issues/5527
- https://arxiv.org/pdf/1411.4389
