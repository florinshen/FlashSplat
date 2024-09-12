#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from os import path as osp
from tqdm import tqdm
from os import makedirs
# import imageio 
import cv2 
from gaussian_renderer import render
from gaussian_renderer import flashsplat_render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from PIL import Image
import numpy as np
import colorsys

import pdb 

import torch
import torch.nn.functional as F

from segment_anything import sam_model_registry, SamPredictor
import cv2


def save_mp4(dir, fps):
    imgpath = dir
    frames = []
    fourcc = cv2.VideoWriter.fourcc(*'mp4v') 
    fps = float(fps)
    for name in sorted(os.listdir(imgpath)):
        img = osp.join(imgpath, name)
        img = cv2.imread(img)
        frames.append(img)

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(os.path.join(dir,'eval.mp4'), fourcc, fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, 
               slackness=0, view_num=-1, obj_num=256, obj_ids=[]):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")


    sam_ckpt_path = "/home/shenqiuhong/Research/segment-anything/notebooks/checkpoint/sam_vit_h_4b8939.pth"
    sam_model = sam_model_registry['vit_h'](checkpoint=sam_ckpt_path).cuda()
    predictor = SamPredictor(sam_model)

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if view_num > 0:
        view_idx = np.linspace(0, len(views) - 1, view_num, dtype=int).tolist()
        views_used = [views[idx] for idx in view_idx]
    else:
        view_num = len(views)
        views_used = views


    # _view_num = 10
    # view_idx = np.linspace(0, len(views) - 1, _view_num, dtype=int).tolist()
    # views_used = [views[idx] for idx in view_idx]

    # pos_list = [(150, 150), (200, 200), (300, 350)]
    # pos_list = [(100, 300), (80, 180)] # the sheep 

    # for the teatime
    # pos_list = [(100, 300), (80, 180), (150, 230), (140, 390), (100, 450)]
    # pos_labels = [1, 1, 1, 1, 1]


    pos_list = [[550, 500], [300, 300], [500, 200], [600, 300], [420, 250], [600, 240], [620, 600], [630, 420]]
    pos_labels = [1] * len(pos_list)

    def get_nearest_point(proj_xy, pos, gs_depth):
        cur_pos = torch.zeros_like(proj_xy)
        cur_pos[0, :], cur_pos[1, :] = pos[0], pos[1]
        distance = (((proj_xy - cur_pos)) ** 2).sum(dim=0)
        near_gsidx = distance.sort()[1][:100]
        gs_depth[gs_depth <= 0]  = 1000. # set invalid depth as extra far.
        selected = gs_depth[near_gsidx].argmin()
        nearest_gsidx = near_gsidx[selected]
        return nearest_gsidx

    with torch.no_grad():
        for idx, view in enumerate(tqdm(views_used, desc="Rendering removal")):
            render_pkg = flashsplat_render(view, gaussians, pipeline, background)
            proj_xy = render_pkg["proj_xy"]
            gs_depth = render_pkg["gs_depth"]
            # for a selected view 
            nearest_gidx = [get_nearest_point(proj_xy, pos, gs_depth) for pos in pos_list]
            nearest_gidx = torch.stack(nearest_gidx)
            break 


    remain_mask = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device=gaussians.get_xyz.device)
    remain_mask[nearest_gidx] = True
    
    obj_removal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "test_xyz")
    os.makedirs(obj_removal_path, exist_ok=True)

    view_prompt_points = []
    rec_idx = nearest_gidx.argsort()

    for idx, view in enumerate(tqdm(views_used, desc="find corresponding points")):
        render_pkg = flashsplat_render(view, gaussians, pipeline, background, used_mask=remain_mask)
        proj_xy = render_pkg["proj_xy"].permute(1, 0)[rec_idx, :]
        view_prompt_points.append(proj_xy)
    
    def check_bound(val, bound):
        if val <= 0 or val > bound:
            return True 
        else:
            return False

    def check_prompts(input_point, input_label, height, width):
        _points = []
        _labels = []
        for point, label in zip(input_point, input_label):
            if check_bound(point[0], width) or check_bound(point[1], height):
                continue 
            else:
                _points.append(point)
                _labels.append(label)
        return np.array(_points), np.array(_labels)

    mask_dir = "/home/shenqiuhong/Downloads/llff-gs/horns/inpaint_object_mask_255"
    makedirs(mask_dir, exist_ok=True)
    for idx, view in enumerate(tqdm(views_used, desc="SAM all views")):
        prompt_points = view_prompt_points[idx]
        gt_image = (view.original_image[0:3, :, :].detach() * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        predictor.set_image(gt_image)
        input_point = prompt_points.cpu().numpy()
        input_label = np.array(pos_labels)
        input_point, input_label = check_prompts(input_point, input_label, view.image_height, view.image_width)
        if input_label.shape[0] > 0:
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            mask = masks[np.argmax(scores)][:, :, None].astype(np.uint8) * 255
        else:
            mask = np.zeros((view.image_height, view.image_width, 1)).astype(np.uint8)
            # view.image_name
        mask_path = os.path.join(mask_dir, '{}.png'.format(view.image_name))
        cv2.imwrite(mask_path, mask)


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, 
                slackness : float, view_num : int, obj_num : int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, 
                        slackness, view_num, obj_num)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, 
                        slackness, view_num, obj_num)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--slackness", default=0.0, type=float)
    parser.add_argument("--view_num", default=10.0, type=int)
    parser.add_argument("--obj_num", default=1, type=int)
    args = get_combined_args(parser)
    print("Generating mask for: " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # args.object_path = args.object_mask

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, 
                args.slackness, args.view_num, args.obj_num)