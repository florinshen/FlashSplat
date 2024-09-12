#! /bin/bash
# export OPENCV_IO_ENABLE_OPENEXR=1
# export CUDA_VISIBLE_DEVICES=0
# python train.py -s /home/shenqiuhong/Downloads/mip360/garden \
#   --model_path ./output/garden-recon-only \
#   --limit_num -1 --port 20001 \
#   --densify_until_iter 10000 \
#   -r 4


# export OPENCV_IO_ENABLE_OPENEXR=1
# export CUDA_VISIBLE_DEVICES=0
# python train.py -s /home/shenqiuhong/Downloads/mip360/room/ \
#   --model_path ./output/room-recon-only \
#   --limit_num -1 --port 20001 \
#   --densify_until_iter 10000 \
#   -r 4

export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=0
python train.py -s /home/shenqiuhong/Downloads/mip360/bonsai/ \
  --model_path ./output/bonsai-recon-only \
  --limit_num -1 --port 20001 \
  --densify_until_iter 10000 \
  -r 4
