#! /bin/bash
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=1
python train.py -s /home/shenqiuhong/dataset/nerf_data/nerf_synthetic/lego \
  --model_path ./output/lego-recon-only \
  --limit_num -1 --port 20009 \
  -r 1
