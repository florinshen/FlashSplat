#! /bin/bash
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=0
python render.py --iteration 30000 \
    -s /home/shenqiuhong/Downloads/Gaussian-Grouping/data/mipnerf360/kitchen/  \
    -m ./output/kitchen-recon-only \
    --skip_test \
    --slackness -0.8 \
    --obj_id 0 \
    --view_num -1 \
    --obj_num 1 \
    --object_path inpaint_object_mask_255 \


# #! /bin/bash
# export OPENCV_IO_ENABLE_OPENEXR=1
# export CUDA_VISIBLE_DEVICES=0
# python render.py --iteration 30000 \
#     -s /home/shenqiuhong/Downloads/Gaussian-Grouping/data/mipnerf360/kitchen/  \
#     -m ./output/kitchen-recon-only \
#     --skip_test \
#     --slackness 0.6 \
#     --view_num -1 \
#     --obj_num 256 \
#     --white_background \
#     # --object_path inpaint_object_mask_255 \