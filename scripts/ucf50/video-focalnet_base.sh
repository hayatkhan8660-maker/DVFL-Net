#!/bin/bash

nvidia-smi
torchrun --nproc_per_node 2 main.py --cfg configs/ucf101/video-focalnet_base.yaml --output output/ --opts DATA.NUM_FRAMES 8