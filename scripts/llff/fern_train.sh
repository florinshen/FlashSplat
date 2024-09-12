
#! /bin/bash
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=1
python train.py -s /home/shenqiuhong/Downloads/llff-gs/flower \
  --model_path ./output/flower-recon-only \
  --limit_num -1 --port 20009 \
  --densify_until_iter 10000 \
  -r 4

#! /bin/bash
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=1
python train.py -s /home/shenqiuhong/Downloads/llff-gs/fortress \
  --model_path ./output/fortress-recon-only \
  --limit_num -1 --port 20009 \
  --densify_until_iter 10000 \
  -r 4

#! /bin/bash
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=1
python train.py -s /home/shenqiuhong/Downloads/llff-gs/trex \
  --model_path ./output/trex-recon-only \
  --limit_num -1 --port 20009 \
  --densify_until_iter 10000 \
  -r 4


