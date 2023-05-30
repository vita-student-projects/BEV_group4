import os
from argparse import ArgumentParser
import argparse

import numpy as np
import torch

import src
from src.utils import MetricDict


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args(notebook=False):
    parser = ArgumentParser()

    # ----------------------------- Data options ---------------------------- #
    parser.add_argument(
        "--root",
        type=str,
        default="/Users/gloriamellinand/Downloads/translating-images-into-maps-main/nuscenes_data",
        help="root directory of the dataset",
    )
    parser.add_argument(
        "--nusc-version", type=str, default="v1.0-mini", help="nuscenes version",
    )
    parser.add_argument(
        "--occ-gt",
        type=str,
        default="200down100up",
        help="occluded (occ) or unoccluded(unocc) ground truth maps",
    )
    parser.add_argument(
        "--gt-version",
        type=str,
        default="semantic_maps_new_200x200",
        help="ground truth name",
    )
    parser.add_argument(
        "--train-split", type=str, default="train_mini", help="ground truth name",
    )
    parser.add_argument(
        "--val-split", type=str, default="val_mini", help="ground truth name",
    )
    parser.add_argument(
        "--data-size",
        type=float,
        default=0.2,
        help="percentage of dataset to train on",
    )
    parser.add_argument(
        "--load-classes-nusc",
        type=str,
        nargs=14,
        default=[
            "drivable_area",
            "ped_crossing",
            "walkway",
            "carpark_area",
            "road_segment",
            "lane",
            "bus",
            "bicycle",
            "car",
            "construction_vehicle",
            "motorcycle",
            "trailer",
            "truck",
            "pedestrian",
            "trafficcone",
            # "barrier",
        ],
        help="Classes to load for NuScenes",
    )
    parser.add_argument(
        "--pred-classes-nusc",
        type=str,
        nargs=12,
        default=[
            "drivable_area",
            "ped_crossing",
            "walkway",
            "carpark_area",
            "bus",
            "bicycle",
            "car",
            "construction_vehicle",
            "motorcycle",
            "trailer",
            "truck",
            "pedestrian",
            "trafficcone",
            # "barrier",
        ],
        help="Classes to predict for NuScenes",
    )
    parser.add_argument(
        "--lidar-ray-mask",
        type=str,
        default="dense",
        help="sparse or dense lidar ray visibility mask",
    )
    parser.add_argument(
        "--grid-size",
        type=float,
        nargs=2,
        default=(50.0, 50.0),
        help="width and depth of validation grid, in meters",
    )
    parser.add_argument(
        "--z-intervals",
        type=float,
        nargs="+",
        default=[1.0, 9.0, 21.0, 39.0, 51.0],
        help="depths at which to predict BEV maps",
    )
    parser.add_argument(
        "--grid-jitter",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="magn. of random noise applied to grid coords",
    )
    parser.add_argument(
        "--aug-image-size",
        type=int,
        nargs="+",
        default=[1280, 720],
        help="size of random image crops during training",
    )
    parser.add_argument(
        "--desired-image-size",
        type=int,
        nargs="+",
        default=[1600, 900],
        help="size images are padded to before passing to network",
    )
    parser.add_argument(
        "--yoffset",
        type=float,
        default=1.74,
        help="vertical offset of the grid from the camera axis",
    )

    # -------------------------- Model options -------------------------- #
    parser.add_argument(
        "--model-name",
        type=str,
        default="PyrOccTranDetr_S_0904_old_rep100x100_out100x100",
        help="Model to train",
    )
    parser.add_argument(
        "-r",
        "--grid-res",
        type=float,
        default=0.5,
        help="size of grid cells, in meters",
    )
    parser.add_argument(
        "--frontend",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50"],
        help="name of frontend ResNet architecture",
    )
    parser.add_argument(
        "--pretrained",
        type=bool,
        default=True,
        help="choose pretrained frontend ResNet",
    )
    parser.add_argument(
        "--pretrained-bem",
        type=bool,
        default=True,
        help="choose pretrained BEV estimation model",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="27_04_23_11_08",
        help="name of pretrained model to load",
    )
    parser.add_argument(
        "--load-ckpt",
        type=str,
        default="checkpoint-epfl-1-0010.pth.gz",
        help="name of checkpoint to load",
    )
    parser.add_argument(
        "--ignore", type=str, default=["nothing"], help="pretrained modules to ignore",
    )
    parser.add_argument(
        "--ignore-reload",
        type=str,
        default=["nothing"],
        help="pretrained modules to ignore",
    )
    parser.add_argument(
        "--focal-length", type=float, default=1266.417, help="focal length",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs=4,
        default=[8.0, 16.0, 32.0, 64.0],
        help="resnet frontend scale factor",
    )
    parser.add_argument(
        "--cropped-height",
        type=float,
        nargs=4,
        default=[20.0, 20.0, 20.0, 20.0],
        help="resnet feature maps cropped height",
    )
    parser.add_argument(
        "--y-crop",
        type=float,
        nargs=4,
        default=[15, 15.0, 15.0, 15.0],
        help="Max y-dimension in world space for all depth intervals",
    )
    parser.add_argument(
        "--dla-norm",
        type=str,
        default="GroupNorm",
        help="Normalisation for inputs to topdown network",
    )
    parser.add_argument(
        "--bevt-linear-additions",
        type=str2bool,
        default=False,
        help="BatchNorm, ReLU and Dropout addition to linear layer in BEVT",
    )
    parser.add_argument(
        "--bevt-conv-additions",
        type=str2bool,
        default=False,
        help="BatchNorm, ReLU and Dropout addition to conv layer in BEVT",
    )
    parser.add_argument(
        "--dla-l1-nchannels",
        type=int,
        default=64,
        help="vertical offset of the grid from the camera axis",
    )
    parser.add_argument(
        "--n-enc-layers",
        type=int,
        default=2,
        help="number of transfomer encoder layers",
    )
    parser.add_argument(
        "--n-dec-layers",
        type=int,
        default=2,
        help="number of transformer decoder layers",
    )

    # ---------------------------- Loss options ---------------------------- #
    parser.add_argument(
        "--loss", type=str, default="dice_loss_mean", help="Loss function",
    )
    parser.add_argument(
        "--exp-cf",
        type=float,
        default=0.0,
        help="Exponential for class frequency in weighted dice loss",
    )
    parser.add_argument(
        "--exp-os",
        type=float,
        default=0.2,
        help="Exponential for object size in weighted dice loss",
    )

    # ------------------------ Optimization options ----------------------- #
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
    parser.add_argument("-l", "--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.99,
        help="factor to decay learning rate by every epoch",
    )

    # ------------------------- Training options ------------------------- #
    parser.add_argument(
        "-e", "--epochs", type=int, default=40, help="number of epochs to train for"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=8, help="mini-batch size for training"
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=5,
        help="Gradient accumulation over number of batches",
    )

    # ------------------------ Experiment options ----------------------- #
    parser.add_argument(
        "--name", type=str,
        default="27_04_23_11_08",
        help="name of experiment",
    )
    parser.add_argument(
        "-s",
        "--savedir",
        type=str,
        default="pretrained_models",
        help="directory to save experiments to",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        nargs="*",
        default=[0],
        help="ids of gpus to train on. Leave empty to use cpu",
    )
    parser.add_argument(
        "--num-gpu", type=int, default=1, help="number of gpus",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=0,
        help="number of worker threads to use for data loading",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=1,
        help="number of epochs between validation runs",
    )
    parser.add_argument(
        "--print-iter",
        type=int,
        default=5,
        help="print loss summary every N iterations",
    )
    parser.add_argument(
        "--vis-iter",
        type=int,
        default=20,
        help="display visualizations every N iterations",
    )
    parser.add_argument(
        "--cuda-available",
        type=int,
        default=0,
        help="defines cuda or cpu environment",
    )
    parser.add_argument(
        "--iou",
        type=int,
        default=1,
        help="defines iou metric to use (0 for iou, 1 for diou)",
    )
    if notebook:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()


def init(args):
    args.savedir = os.path.join(os.getcwd(), args.savedir)

    # Build depth intervals along Z axis and reverse
    z_range = args.z_intervals
    args.grid_size = (z_range[-1] - z_range[0], z_range[-1] - z_range[0])

    # Calculate cropped heights of feature maps
    h_cropped = src.utils.calc_cropped_heights(
        args.focal_length, np.array(args.y_crop), z_range, args.scales
    )
    args.cropped_height = [h for h in h_cropped]

    num_gpus = torch.cuda.device_count()
    args.num_gpu = num_gpus
