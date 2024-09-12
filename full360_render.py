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
import cv2 
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.cameras import Camera


import numpy as np

import pdb

def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)
    
def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))
    
def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)

def look_at(campos, target):
    forward_vector = safe_normalize(campos - target)
    world_up = np.array([0, -1, 0], dtype=np.float32)
    # right_vector = safe_normalize(np.cross(up_vector, forward_vector))
    right_vector = safe_normalize(np.cross(world_up, forward_vector))
    up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R

def pose_spherical(elevation, azimuth, radius=4.031, opengl=False):
    elevation = np.deg2rad(elevation)
    azimuth = np.deg2rad(azimuth)

    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = -radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.sin(azimuth)
    campos = np.array([x, y, z])
    c2w = np.eye(4, dtype=np.float32)
    target = np.zeros([3], dtype=np.float32)
    c2w[:3, :3] = look_at(campos, target)
    c2w[:3, 3] = campos # raw format of c2w in transform.json

    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    # c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]
    return R, T


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


def adjust_pose(view):
    R, T = view.R.copy(), view.T.copy()
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, -1] = T
    c2w = np.linalg.inv(w2c)

    cam_T = c2w[:3, -1].copy()
    cam_R = c2w[:3, :3].copy()
    # cam_T[2] += 15
    cam_T[1] += 15

    c2w = np.eye(4)
    c2w[:3, :3] = cam_R
    c2w[:3, -1] = cam_T
    new_w2c = np.linalg.inv(c2w)
    new_R = new_w2c[:3, :3] 
    new_T = new_w2c[:3, -1]

    view.R = new_R 
    view.T = new_T

    pdb.set_trace()
    return view

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, 'webpage', name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, 'webpage', name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    positions = gaussians.get_xyz.mean(dim=0)

    render_poses = [pose_spherical(azimuth=angle, elevation=0., radius=15.0) for angle in np.linspace(-180,180,120+1)[:-1]]
    render_cams = []
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    train_viewpoint = views.pop(0)
    # for idx, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
    #     R, T = pose
    #     cam = Camera(colmap_id=idx, R=R, T=T, 
    #                 FoVx=train_viewpoint.FoVx, FoVy=train_viewpoint.FoVy, 
    #                 image=train_viewpoint.original_image, gt_alpha_mask=None, gt_depth=None,
    #                 image_name="", uid=0, data_device=train_viewpoint.data_device)
    #     rendering = render(cam, gaussians, pipeline, background)["render"]
    #     torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # R, T = pose
        # cam = Camera(colmap_id=idx, R=R, T=T, 
        #             FoVx=train_viewpoint.FoVx, FoVy=train_viewpoint.FoVy, 
        #             image=train_viewpoint.original_image, gt_alpha_mask=None, gt_depth=None,
        #             image_name="", uid=0, data_device=train_viewpoint.data_device)

        view = adjust_pose(view)
        rendering = render(view, gaussians, pipeline, background)["render"]
        pdb.set_trace()
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

    save_mp4(render_path, 15)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)