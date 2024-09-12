#! /bin/bash
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=0
python train.py -s /home/shenqiuhong/Downloads/Gaussian-Grouping/data/mipnerf360/kitchen/ \
  --model_path ./output/kitchen-recon-only \
  --limit_num -1 --port 20009 \
  --densify_until_iter 10000 \
  -r 2
