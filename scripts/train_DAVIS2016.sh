#!/bin/bash

# Script to Train a model on the DAVIS 2016 dataset (https://davischallenge.org/index.html)

python3 ../train.py \
--dataset=DAVIS2016 \
--root_dir='../images' \
--test_temporal_shift=1 \
--checkpoint_dir=../ckpt/autoencoder
