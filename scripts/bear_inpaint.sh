export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=0
python edit_object_inpaint.py --iteration 30000 \
    -s /home/shenqiuhong/Downloads/Gaussian-Grouping/data/bear/ \
    --image images_inpaint_unseen \
    -m ./output/bear-recon-only/ \
    --skip_test \
    --view_num -1 \
    --obj_num 1 \
    --white_background \
    --object_path inpaint_object_mask_255 \
    --finetune_iteration 10000 \
    --obj_id 1 \