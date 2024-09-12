python render.py --iteration 30000 \
    -s /home/shenqiuhong/Downloads/Gaussian-Grouping/data/lerf_mask/teatime/ \
    -m ./output/teatime-recon-only/ \
    --skip_test \
    --slackness -0.9 \
    --view_num -1 \
    --obj_num 1 \
    --white_background \
    --obj_id 1 \
    --object_path inpaint_object_mask_255

# python render.py --iteration 30000 \
#     -s /home/shenqiuhong/Downloads/Gaussian-Grouping/data/bear/ \
#     -m ./output/bear-recon-only/ \
#     --skip_test \
#     --slackness -0.6 \
#     --view_num -1 \
#     --obj_num 1 \
#     --white_background \
#     # --object_path inpaint_object_mask_255 \