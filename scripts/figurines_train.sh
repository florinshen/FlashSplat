#! /bin/bash
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=0
python train.py -s /home/shenqiuhong/Downloads/Gaussian-Grouping/data/lerf_mask/figurines/ \
  --model_path ./output/figurines-recon-only2 \
  --limit_num -1 --port 20001 \
  --densify_until_iter 10000 \