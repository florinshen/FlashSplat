python colormask.py --iteration 30000 \
    -s /home/shenqiuhong/Downloads/Gaussian-Grouping/data/mipnerf360/counter \
    -m ./output/counter-recon-only/ \
    --skip_test \
    --slackness 0.88 \
    --view_num -1 \
    --obj_num 256 \
    --white_background \
    -r 2
    # --object_path inpaint_object_mask_255 \