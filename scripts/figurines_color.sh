#! /bin/bash
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=0
python colormask.py --iteration 30000 \
    -s /home/shenqiuhong/Downloads/Gaussian-Grouping/data/lerf_mask/figurines \
    -m ./output/figurines-recon-only2/ \
    --skip_test \
    --slackness 0.8 \
    --view_num -1 \
    --obj_num 256 \
    --white_background \
    # --object_path inpaint_object_mask_255 \