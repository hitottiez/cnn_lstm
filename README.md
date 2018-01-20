# Overview
This code is for action recognition using the two stream networks (rgb and optical flow).
Each stream is constructed by CNN and LSTM.
I implemented this code for practice because I have not written any deep learning code from scratch!

# Setup
## Environment
- Linux (I used Ubuntu 14.04)
- GPU (I used GTX TitanX x2)
- Keras + TensorFlow (I recommend to install Keras via docker)

## Dataset
Download [UCF-101 dataset](http://crcv.ucf.edu/data/UCF101.php), and set to arbitary directory.

# Run
## Convert video to RGB images or optical flows
Use scripts/extract_optical_flow.sh in [TSN (ECCV2016)](https://github.com/yjxiong/temporal-segment-networks).

## Make train/val split file
Use scripts/build_file_list.sh in [TSN (ECCV2016)](https://github.com/yjxiong/temporal-segment-networks).

## Train model and test it
You can train a model and test it.
```Shell
bash run.sh
```

# TODO
- Modify to accept the various length input sequences
(Fixed number of frames are extracted from a sequence now)

# Reference
## CNN + LSTM
- [Keras issue 401](https://github.com/keras-team/keras/issues/401)
- [Keras issue 5527](https://github.com/keras-team/keras/issues/5527)
- [LRCN (CVPR2015)](https://arxiv.org/pdf/1411.4389)

## Batch training using "fit_generator" method
- [Hatena blog](http://hironsan.hatenablog.com/entry/2017/09/09/130608)

## Two stream CNN
- [Two stream CNN (NIPS2014)](https://arxiv.org/pdf/1406.2199)
