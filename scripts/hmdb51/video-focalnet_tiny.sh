#!/bin/bash

nvidia-smi
torchrun --nproc_per_node 2 main.py --cfg configs/hmdb51/video-focalnet_tiny.yaml --output output/ --opts DATA.NUM_FRAMES 8