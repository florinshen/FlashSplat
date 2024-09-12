# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import cv2
from utils.loss_utils import masked_l1_loss
from random import randint
import lpips
import json

import pdb 


def mask_to_bbox(mask):
    # Find the rows and columns where the mask is non-zero
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]
    
    return xmin, ymin, xmax, ymax

def crop_using_bbox(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    return image[:, ymin:ymax+1, xmin:xmax+1]

# Function to divide image into K x K patches
def divide_into_patches(image, K):
    B, C, H, W = image.shape
    patch_h, patch_w = H // K, W // K
    patches = torch.nn.functional.unfold(image, (patch_h, patch_w), stride=(patch_h, patch_w))
    patches = patches.view(B, C, patch_h, patch_w, -1)
    return patches.permute(0, 4, 1, 2, 3)

def finetune_inpaint(opt, model_path, iteration, views, gaussians, pipeline, background, selected_obj_ids, cameras_extent, finetune_iteration, view_num, obj_num):

    stats_counts_path = os.path.join(model_path, 'train', "ours_{}".format(iteration), "used_count")
    makedirs(stats_counts_path, exist_ok=True)
    if view_num < 0:
        view_num = len(views)
    cur_count_dir = os.path.join(stats_counts_path, "view_num_{:03d}_objnum_{:03d}.pth".format(view_num, obj_num))

    from render import multi_instance_opt
    all_counts = torch.load(cur_count_dir).cuda()
    all_obj_labels = multi_instance_opt(all_counts, -0.4)
    mask3d = (all_obj_labels[selected_obj_ids].bool()) # remained gs, as true
    # pdb.set_trace()

    # fix some gaussians
    gaussians.inpaint_setup(opt, mask3d)
    print()
    iterations = finetune_iteration
    progress_bar = tqdm(range(iterations), desc="Finetuning progress")
    LPIPS = lpips.LPIPS(net='vgg')
    for param in LPIPS.parameters():
        param.requires_grad = False
    LPIPS.cuda()


    for iteration in range(iterations):
        viewpoint_stack = views.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        render_pkg = render(viewpoint_cam, gaussians, pipeline, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        mask2d = viewpoint_cam.objects > 128
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = masked_l1_loss(image, gt_image, ~mask2d)

        bbox = mask_to_bbox(mask2d)
        cropped_image = crop_using_bbox(image, bbox)
        cropped_gt_image = crop_using_bbox(gt_image, bbox)
        K = 2
        rendering_patches = divide_into_patches(cropped_image[None, ...], K)
        gt_patches = divide_into_patches(cropped_gt_image[None, ...], K)
        lpips_loss = LPIPS(rendering_patches.squeeze()*2-1,gt_patches.squeeze()*2-1).mean()

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * lpips_loss
        loss.backward()

        with torch.no_grad():
            if iteration < 5000 :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if  iteration % 300 == 0:
                    size_threshold = 20 
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, cameras_extent, size_threshold)
                
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none = True)

        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
            progress_bar.update(10)
    progress_bar.close()

    # save gaussians
    point_cloud_path = os.path.join(model_path, "point_cloud_object_inpaint/iteration_{}".format(iteration))
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    return gaussians




def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours{}".format(iteration), "gt")
    colormask_path = os.path.join(model_path, name, "ours{}".format(iteration), "objects_feature16")
    gt_colormask_path = os.path.join(model_path, name, "ours{}".format(iteration), "gt_objects_color")
    pred_obj_path = os.path.join(model_path, name, "ours{}".format(iteration), "objects_pred")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(colormask_path, exist_ok=True)
    makedirs(gt_colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    from render import save_mp4
    save_mp4(render_path, 15)



def inpaint(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, opt : OptimizationParams, select_obj_id : int, finetune_iteration: int,
            view_num : int, obj_num : int):
    # 1. load gaussian checkpoint
    gaussians = GaussianModel(dataset.sh_degree)
    # scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    # # # 2. inpaint selected object
    # gaussians = finetune_inpaint(opt, dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, select_obj_id, scene.cameras_extent, finetune_iteration,
    #                              view_num, obj_num)

    # 3. render new result
    dataset.object_path = 'object_mask'
    dataset.images = 'images'
    scene = Scene(dataset, gaussians, load_iteration='_object_inpaint/iteration_'+str(finetune_iteration-1), shuffle=False)
    with torch.no_grad():
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--config_file", type=str, default="config/object_removal/bear.json", help="Path to the configuration file")

    parser.add_argument("--slackness", default=0.0, type=float)
    parser.add_argument("--view_num", default=10.0, type=int)
    parser.add_argument("--obj_num", default=1, type=int)
    parser.add_argument("--obj_id", default=-1, type=int)

    parser.add_argument("--finetune_iteration", default=10_000, type=int)


    args = get_combined_args(parser)
    print("Rendering " + args.model_path)




    # args.num_classes = config.get("num_classes", 200)
    # args.removal_thresh = config.get("removal_thresh", 0.3)
    # args.select_obj_id = config.get("select_obj_id", [34])
    # args.images = config.get("images", "images")
    # args.object_path = config.get("object_path", "object_mask")
    # args.resolution = config.get("r", 1)
    # args.lambda_dssim = config.get("lambda_dlpips", 0.5)
    # args.finetune_iteration = config.get("finetune_iteration", 10_000)
    assert args.obj_id > 0
    select_obj_id = args.obj_id
    # Initialize system state (RNG)
    safe_state(args.quiet)

    inpaint(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, opt.extract(args), select_obj_id, args.finetune_iteration,
            args.view_num, args.obj_num)