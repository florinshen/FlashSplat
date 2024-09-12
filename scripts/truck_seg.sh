#! /bin/bash
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=1
python render.py --iteration 30000 \
    -s /home/shenqiuhong/Downloads/tandt/tandt/truck \
    -m ./output/truck-recon-only/ \
    --skip_test \
    --slackness 0.7 \
    --view_num -1 \
    --obj_num 1 \
    --white_background \
    --object_path inpaint_object_mask_255 \