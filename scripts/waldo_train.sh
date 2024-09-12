#! /bin/bash
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=1
python train.py -s /home/shenqiuhong/Downloads/lerf_dataset/waldo_kitchen \
  --model_path ./output/waldo_kitchen-recon-only \
  --limit_num -1 --port 20004 \
  --densify_until_iter 10000 \