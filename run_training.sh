#!/bin/bash
set -e

python prepare.py
cd detector
eps=100
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python main.py --model res18 -b 2 --epochs $eps --save-dir res18 # --save-freq 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python main.py --model res18 -b 2 --resume results/res18/$eps.ckpt --test 1
cp results/res18/$eps.ckpt ../../model/detector.ckpt
