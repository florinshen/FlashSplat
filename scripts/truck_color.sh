#! /bin/bash
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=1
python colormask.py --iteration 30000 \
    -s  /home/shenqiuhong/Downloads/tandt/tandt/truck \
    -m ./output/truck-recon-only/ \
    --skip_test \
    --slackness 0.6 \
    --view_num -1 \
    --obj_num 1 \
    --white_background \
    -r 2