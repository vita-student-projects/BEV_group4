import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import cv2

from pathlib import Path
from collections import namedtuple, Counter
import itertools

import torch
import torch.nn.functional as F

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility

from src.data.dataloader import nuScenesMaps, read_split
from argslib import parse_args, init

color_map_labels = {
    'drivable_area': np.array([160, 160, 160]),
    'ped_crossing': np.array([0, 153, 0]),
    'walkway': np.array([0, 102, 204]),
    'carpark_area': np.array([96, 96, 96]),
    'bus': np.array([255, 128, 0]),
    'bicycle': np.array([255, 55, 255]),
    'car': np.array([255, 255, 51]),
    'construction_vehicle': np.array([153, 0, 76]),
    'motorcycle': np.array([255, 55, 255]),
    'trailer': np.array([204, 0, 0]),
    'truck': np.array([255, 102, 102]),
    'pedestrian': np.array([51, 51, 255]),
    'trafficcone': np.array([255, 204, 153])
    # 'trafficcone': np.array([255, 204, 153]),
    # 'barrier': np.array([64, 64, 64])
}
COLOR_MAP = {(i + 1): color for i, color in enumerate(color_map_labels.values())}


def color_components(labels, color_map=COLOR_MAP):
    """
    label 0 is assigned white color to have a white background.

    Iterates through the image to replace each pixel with the color associated with its label.

    Returns the colored image.
    """
    colors = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    n_rows, n_cols = labels.shape
    for i in range(n_rows):
        for j in range(n_cols):
            colors[i, j, :] = color_map.get(labels[i, j], np.array([255, 255, 255]))

    return colors


def make_composite(cls_maps):
    """
    cls_maps: Nc x h x w - boolean masks tensors
    Nc - number of classes - len(args.pred_classes_nusc)

    Output: h x w tensors where each pixel will have a class index
    """
    nc = cls_maps.shape[0]
    class_idx = torch.arange(nc) + 1
    x = (cls_maps > 0.5).float() * class_idx.view(-1, 1, 1)
    cls_map_composite, _ = x.max(dim=0)
    return cls_map_composite


def plot_ground_truth(gt_image, cls_maps, out_path=None, dpi=300):
    composite = make_composite(cls_maps)
    cls_maps_colors = cv2.flip(color_components(composite.numpy()), 0)
    gt_image_np = gt_image.numpy().transpose((1, 2, 0))  # (3. 900, 1600) -> (1600, 900, 3)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

    ax = axs[0]
    ax.imshow(gt_image_np)
    ax.set_title('CAM_FRONT')
    ax.axis('off')

    ax = axs[1]
    ax.imshow(cls_maps_colors)
    ax.set_title('BEV - ground truth')
    legend_colors = [np.append(c / 255, 1) for c in COLOR_MAP.values()]
    patches = [mpatches.Patch(color=legend_colors[i], label=label)
               for i, label in enumerate(color_map_labels.keys())]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.axis('off')

    # access each axes object via axs.flat
    for ax in axs.flat:
        # check if something was plotted
        if not bool(ax.has_data()):
            fig.delaxes(ax)  # delete if nothing is plotted in the axes obj

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=dpi)
    else:
        plt.show()


def visualize_prediction(gt_image, cls_maps, pred, out_path=None, figsize=(15, 7), dpi=300):
    composite = make_composite(cls_maps)
    cls_maps_colors = cv2.flip(color_components(composite.numpy()), 0)
    gt_image_np = gt_image.numpy().transpose((1, 2, 0))  # (3. 900, 1600) -> (1600, 900, 3)

    composite_pred = make_composite(pred)
    pred_colors = cv2.flip(color_components(composite_pred.numpy()), 0)
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize)

    ax = axs[0]
    ax.imshow(gt_image_np)
    ax.set_title('CAM_FRONT')
    ax.axis('off')

    ax = axs[1]
    ax.imshow(cls_maps_colors)
    ax.set_title('BEV - ground truth')
    ax.axis('off')

    ax = axs[2]
    ax.imshow(pred_colors)
    ax.set_title('BEV - prediction')
    legend_colors = [np.append(c / 255, 1) for c in COLOR_MAP.values()]
    patches = [mpatches.Patch(color=legend_colors[i], label=label)
               for i, label in enumerate(color_map_labels.keys())]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.axis('off')

    # access each axes object via axs.flat
    for ax in axs.flat:
        # check if something was plotted
        if not bool(ax.has_data()):
            fig.delaxes(ax)  # delete if nothing is plotted in the axes obj

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=dpi)
    else:
        plt.show()


def main():
    args = parse_args(notebook=False)
    init(args)

    train_data = nuScenesMaps(
        root=args.root,
        split=args.train_split,
        grid_size=args.grid_size,
        grid_res=args.grid_res,
        classes=args.load_classes_nusc,
        dataset_size=args.data_size,
        desired_image_size=args.desired_image_size,
        mini=True,
        gt_out_size=(100, 100),
    )
    train_tokens = read_split(
        os.path.join(args.root, "splits", "{}.txt".format(args.train_split))
    )
    val_data = nuScenesMaps(
        root=args.root,
        split=args.val_split,
        grid_size=args.grid_size,
        grid_res=args.grid_res,
        classes=args.load_classes_nusc,
        dataset_size=args.data_size,
        desired_image_size=args.desired_image_size,
        mini=True,
        gt_out_size=(100, 100),
    )
    val_tokens = read_split(
        os.path.join(args.root, "splits", "{}.txt".format(args.val_split))
    )

    image, cls_maps, vis_mask, calib, grid2d = val_data.__getitem__(0)

    plot_ground_truth(image, cls_maps)


if __name__ == "__main__":
    main()
