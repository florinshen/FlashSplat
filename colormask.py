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

def mean_neighborhood(input_img, N):

    pad = (N - 1) // 2
    padded_img = F.pad(input_img, (pad, pad, pad, pad), mode='constant', value=0)
    patches = padded_img.unfold(1, N, 1).unfold(2, N, 1)
    mean_patches = patches.mean(dim=-1).mean(dim=-1)
    return mean_patches




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


# def multi_instance_opt(all_counts, slackness=0.):
#     all_counts = torch.nn.functional.normalize(all_counts, dim=0)

#     all_counts_sum = all_counts.sum(dim=0)

#     all_obj_labels = torch.zeros_like(all_counts)
#     obj_num = all_counts.size(0)
#     for obj_idx, obj_counts in enumerate(tqdm(all_counts, desc="multi-view optimize")):
#         if obj_counts.sum().item() == 0:
#             continue        
#         # other_idx = list(range(obj_idx)) + list(range(obj_idx + 1, obj_num))
#         # other_counts = all_counts[other_idx, :].sum(dim=0)
#         # obj_counts = torch.stack([other_counts, obj_counts], dim=0)
#         # dynamic programming
#         obj_counts = torch.stack([all_counts_sum - obj_counts, obj_counts], dim=0)
#         if slackness != 0:
#             obj_counts = torch.nn.functional.normalize(obj_counts, dim=0)
#             obj_counts[0, :] += slackness
#         obj_label = obj_counts.max(dim=0)[1]
#         all_obj_labels[obj_idx] = obj_label
#     return all_obj_labels


import torch.nn.functional as F
def multi_instance_opt(all_contrib, gamma=0.):
    """
    Input:
    all_contrib: A_{e} with shape (obj_num, gs_num) 
    gamma: softening factor range from [-1, 1]
    
    Output: 
    all_obj_labels: results S with shape (obj_num, gs_num)
    where S_{i,j} denotes j-th gaussian belong i-th object
    """
    all_contrib_sum = all_contrib.sum(dim=0)
    all_obj_labels = torch.zeros_like(all_contrib).bool()
    for obj_idx, obj_contrib in enumerate(all_contrib, desc="multi-view optimize"):
        obj_contrib = torch.stack([all_contrib_sum - obj_contrib, obj_contrib], dim=0)
        obj_contrib = F.normalize(obj_contrib, dim=0)
        obj_contrib[0, :] += gamma
        obj_label = torch.argmax(obj_contrib, dim=0)
        all_obj_labels[obj_idx] = obj_label
    return all_obj_labels

def id2rgb(id, max_num_obj=256):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)           # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5       # Alternate between 0.5 and 1.0
    l = 0.5

    
    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.uint8)
    if id==0:   #invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb

# for visualize color mask
def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, 
               slackness=0, view_num=-1, obj_num=256, obj_id=-1):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if view_num > 0:
        view_idx = np.linspace(0, len(views) - 1, view_num, dtype=int).tolist()
        views_used = [views[idx] for idx in view_idx]
    else:
        view_num = len(views)
        views_used = views

    all_counts = None
    stats_counts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "used_count")
    makedirs(stats_counts_path, exist_ok=True)
    cur_count_dir = os.path.join(stats_counts_path, "view_num_{:03d}_objnum_{:03d}.pth".format(view_num, obj_num))
    if os.path.exists(cur_count_dir):
        print(f"find {cur_count_dir}")
        all_counts = torch.load(cur_count_dir).cuda()
    else:
        for idx, view in enumerate(tqdm(views_used, desc="Rendering progress")):
            if obj_num == 1:
                gt_mask = view.objects.to(torch.float32) / 255.
            else:
                gt_mask = view.objects.to(torch.float32)
                assert torch.any(torch.max(gt_mask) < obj_num), f"max_obj {int(torch.max(gt_mask).item())}"
            # set(tuple(gt_mask.cpu().tolist())[0])
            render_pkg = flashsplat_render(view, gaussians, pipeline, background, gt_mask=gt_mask, obj_num=obj_num)
            rendering = render_pkg["render"]
            used_count = render_pkg["used_count"]
            if all_counts is None:
                all_counts = torch.zeros_like(used_count)
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            all_counts += used_count
        save_mp4(render_path, 15)
        torch.save(all_counts, cur_count_dir)

    if all_counts is not None:

        # pdb.set_trace()
        # cup_counts = all_counts[39:40, :]
        # all_counts = torch.cat([all_counts.sum(dim=0, keepdim=True) - cup_counts, cup_counts], dim=0)

        all_obj_labels = multi_instance_opt(all_counts, slackness)
             
        render_num = all_counts.size(0)

        # view_interval = (len(views) // view_num + 1)
        # views = views[::view_interval]
        # _view_num = 10
        # view_idx = np.linspace(0, len(views) - 1, _view_num, dtype=int).tolist()
        # views_used = [views[idx] for idx in view_idx]
        # another option, render as color mask
        color_mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "color_mask")
        makedirs(color_mask_path, exist_ok=True)
        for idx, view in enumerate(tqdm(views, desc="Rendering color mask")):
            pred_mask = None
            max_alpha = None
            min_depth = None
            for obj_idx in range(render_num):
                obj_used_mask = (all_obj_labels[obj_idx]).bool()
                if obj_used_mask.sum().item() == 0:
                    continue
                render_pkg = flashsplat_render(view, gaussians, pipeline, background, used_mask=obj_used_mask)
                render_alpha = render_pkg["alpha"]
                render_depth = render_pkg["depth"]
                if pred_mask is None:
                    pred_mask = torch.zeros_like(render_alpha)
                    max_alpha = torch.zeros_like(render_alpha)
                    min_depth = torch.ones_like(render_alpha)
                
                # pix_mask = (render_alpha > max_alpha)
                # pred_mask[pix_mask] = obj_idx

                _pix_mask = (render_alpha > 0.5)
                pix_mask = _pix_mask.clone() 

                overlap_mask = (_pix_mask & (pred_mask > 0)) # shape (1, h, w)

                if overlap_mask.sum().item() > 0:
                    # may use more accurate ctrl here, 3dgs depth acc is limited, deprecated.
                    # if min_depth[overlap_mask].mean() < render_depth[overlap_mask].mean():
                    #     tmp = (min_depth[overlap_mask] < render_depth[overlap_mask])
                    #     pix_mask[overlap_mask] = ~tmp  # set pixel already has label as False, 

                    # if min_depth[overlap_mask].mean() < render_depth[overlap_mask].mean():

                    if (min_depth[overlap_mask].mean() < render_depth[overlap_mask].mean()):
                        # if render_alpha[overlap_mask].mean() > max_alpha[overlap_mask].mean():
                        pix_mask[_pix_mask] = (~(pred_mask[_pix_mask] > 0)) # set pixel already has label as False, 
                        # pix_mask[overlap_mask] = False
                    
                pred_mask[pix_mask] = obj_idx
                # record current depth and alpha
                min_depth[pix_mask] = render_depth[pix_mask]
                max_alpha[pix_mask] = render_alpha[pix_mask]
                
            # color_mask = visualize_obj(pred_mask[0].cpu().numpy().astype(np.uint8))

            vis_mask = pred_mask.repeat(3, 1, 1)
            gt = view.original_image[0:3, :, :]
            gt_mask = vis_mask
            dummy_gt_mask = torch.zeros_like(gt)
            dummy_gt_mask[0] = 1.
            gt[gt_mask.bool()] = 0.7 * gt[gt_mask.bool()] + 0.3 * dummy_gt_mask[gt_mask.bool()]
            torchvision.utils.save_image(gt, os.path.join(color_mask_path, '{0:05d}'.format(idx) + ".png"))
            
            # pdb.set_trace()
            # Image.fromarray(color_mask).save(os.path.join(color_mask_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, 
                slackness : float, view_num : int, obj_num : int, obj_id: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, 
                        slackness, view_num, obj_num, obj_id)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, 
                        slackness, view_num, obj_num, obj_id)

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
    parser.add_argument("--obj_id", default=-1, type=int)
    # parser.add_argument("--object_path", default="inpaint_object_mask_255", type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # args.object_path = args.object_mask

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, 
                args.slackness, args.view_num, args.obj_num, args.obj_id)