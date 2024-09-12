#! /bin/bash
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=0
python objremoval.py --iteration 30000 \
    -s /home/shenqiuhong/Downloads/Gaussian-Grouping/data/mipnerf360/counter \
    -m ./output/counter-recon-only/  \
    --skip_test \
    --slackness -0.9 \
    --view_num -1 \
    --obj_num 256 \
    --obj_id 4 39 45 54 102 \
    -r 2 \
    # --white_background \

