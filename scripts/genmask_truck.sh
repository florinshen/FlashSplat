
#! /bin/bash
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=1
python generate_mask.py --iteration 30000 \
    -s /home/shenqiuhong/Downloads/tandt/tandt/truck \
    -m ./output/truck-recon-only \
    --skip_test \
    --slackness 0 \
    --view_num -1 \
    --obj_num 256 \
    --white_background \