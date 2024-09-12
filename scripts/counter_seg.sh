#! /bin/bash
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=0
python render.py --iteration 30000 \
    -s /home/shenqiuhong/Downloads/Gaussian-Grouping/data/mipnerf360/counter/  \
    -m ./output/counter-recon-only \
    --skip_test \
    --slackness 0.4 \
    --view_num 1 \
    --obj_num 256 \
    --white_background \
    --obj_id 4 \
    # --object_path inpaint_object_mask_255 \