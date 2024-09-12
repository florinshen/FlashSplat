#! /bin/bash
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=0
python train.py -s /home/shenqiuhong/Downloads/lerf_dataset/teatime \
  --model_path ./output/teatime-recon-only \
  --limit_num -1 --port 20003 \
  --densify_until_iter 10000 \