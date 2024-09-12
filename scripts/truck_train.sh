
#! /bin/bash
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=1
python train.py -s /home/shenqiuhong/Downloads/tandt/tandt/truck \
  --model_path ./output/truck-recon-only \
  --limit_num -1 --port 20009 \
  --densify_until_iter 10000 \