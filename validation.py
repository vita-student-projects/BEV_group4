import argparse
import json
import os
import time
from argparse import ArgumentParser
from collections import OrderedDict
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib as mpl
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToPILImage

import src
import src.data.collate_funcs
import src.model.network as networks
from src.data.dataloader import nuScenesMaps
from src.utils import MetricDict

import cv2


def visualize_score(scores,  heatmaps, grid, image, iou, iou_dict, num_classes, args):
    
    # Condense scores and ground truths to single map
    class_idx = torch.arange(len(scores)) + 1
    logits = scores.clone().cpu() * class_idx.view(-1, 1, 1)
    logits, _ = logits.max(dim=0)

    scores = (scores.detach().clone().cpu()>0.5).float() * class_idx.view(-1, 1, 1)
    cls_idx = scores.clone()
    cls_idx = cls_idx.argmax(dim=0)
    cls_idx = cls_idx.numpy()
    color_codes = cv2.applyColorMap(np.uint8(cls_idx * (255/num_classes)), cv2.COLORMAP_JET)
    color_codes = cv2.cvtColor(color_codes, cv2.COLOR_BGR2RGB)

    scores, _ = scores.max(dim=0)
    heatmaps = (heatmaps.detach().clone().cpu()>0.5).float() * class_idx.view(
        -1, 1, 1
    )

    heatmaps, _ = heatmaps.max(dim=0)

    # Visualize score
    fig = plt.figure(num="score", figsize=(10, 8))
    fig.clear()

    # Figure layout in the saved image (including input image, model output logits, predictions, groundtruth)
    gs = mpl.gridspec.GridSpec(2, 4, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])     # first line, all line
    ax2 = fig.add_subplot(gs[1, 0])     # second line, first item
    ax3 = fig.add_subplot(gs[1:, 1])    # second line, second item
    ax4 = fig.add_subplot(gs[1:, 2])    # second line, third item
    ax5 = fig.add_subplot(gs[1:, 3])    # second line, forth item

    image = ax1.imshow(image)
    ax1.grid(which="both")
    image2 = ax2.imshow(color_codes, origin='lower')
    image3 = ax3.imshow(scores, origin='lower', cmap='jet')
    image4 = ax4.imshow(heatmaps, origin='lower', cmap='jet')

    ax5.axis('off')
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=0, vmax=num_classes-1)

    class_names = [args.pred_classes_nusc[i] for i in range(num_classes)]
    handles = [mpl.patches.Patch(color=cmap(norm(i)), label=f"({iou_dict[i]:.2f}) {class_names[i]}") for i in range(num_classes)]
    ax5.legend(handles=handles, loc='center', bbox_to_anchor=(0.5, 0.5))

    grid = grid.cpu().detach().numpy()
    yrange = np.arange(grid[:, 0].max(), step=5)
    xrange = np.arange(start=grid[0, :].min(), stop=grid[0, :].max(), step=5)
    ymin, ymax = 0, grid[:, 0].max()
    xmin, xmax = grid[0, :].min(), grid[0, :].max()

    x2 = plt.vlines(xrange, ymin, ymax, color="white", linewidth=0.5)
    x2 = plt.hlines(yrange, xmin, xmax, color="white", linewidth=0.5)
    x3 = plt.vlines(xrange, ymin, ymax, color="white", linewidth=0.5)
    x3 = plt.hlines(yrange, xmin, xmax, color="white", linewidth=0.5)
    x4 = plt.vlines(xrange, ymin, ymax, color="white", linewidth=0.5)
    x4 = plt.hlines(yrange, xmin, xmax, color="white", linewidth=0.5)

    ax1.set_title("Input image", size=20)
    ax2.set_title("Model output logits", size=10)
    ax3.set_title("Model prediction = logits" + r"$ > 0.5$", size=10)
    ax4.set_title("Ground truth", size=10)
    
    if (args.iou == 1):
        ax5.set_title("DIoU and color code per class", size=10)
        plt.suptitle(
            "DIoU : {:.2f}".format(iou), size=14,
        )
    elif (args.iou == 0):
        ax5.set_title("IoU and color code per class", size=10)
        plt.suptitle(
            "IoU : {:.2f}".format(iou), size=14,
        )
    gs.tight_layout(fig)
    gs.update(top=0.9)

    return fig


def validate(args, dataloader, model, epoch=0):
    print("\n==> Validating on {} minibatches\n".format(len(dataloader)))
    model.eval()
    epoch_loss = MetricDict()
    epoch_iou = MetricDict()
    epoch_loss_per_class = MetricDict()
    num_classes = len(args.pred_classes_nusc)
    iou_mean = np.zeros(num_classes)
    t = time.perf_counter()

    for i, ((image, calib, grid2d), (cls_map, vis_mask)) in enumerate(dataloader):
        if args.cuda_available:
            # Move tensors to GPU
            image, calib, cls_map, vis_mask, grid2d = (
                image.cuda(),
                calib.cuda(),
                cls_map.cuda(),
                vis_mask.cuda(),
                grid2d.cuda(),
            )

        with torch.no_grad():
            # Run network forwards
            pred_ms = model(image, calib, grid2d)

            # Upsample largest prediction to 200x200
            pred_200x200 = F.interpolate(
                pred_ms[0], size=(200, 200), mode="bilinear"
            )
            # pred_200x200 = (pred_200x200 > 0).float()
            pred_ms = [pred_200x200, *pred_ms]

            # Get required gt output sizes
            map_sizes = [pred.shape[-2:] for pred in pred_ms]

            # Convert ground truth to binary mask
            gt_s1 = (cls_map > 0).float()
            vis_mask_s1 = (vis_mask > 0.5).float()

            # Downsample to match model outputs
            gt_ms = src.utils.downsample_gt(gt_s1, map_sizes)
            vis_ms = src.utils.downsample_gt(vis_mask_s1, map_sizes)

            # Compute IoU or dIoU (based on diou variable)
            iou_per_sample, iou_dict = src.utils.compute_multiscale_iou(
                pred_ms, gt_ms, vis_ms, num_classes, args.iou
            )

            # Compute per class loss for eval
            per_class_loss_dict = src.utils.compute_multiscale_loss_per_class(
                pred_ms, gt_ms,
            )

            epoch_iou += iou_dict
            epoch_loss_per_class += per_class_loss_dict

            # Print summary
            batch_time = (time.perf_counter() - t) / (1 if i == 0 else args.accumulation_steps)
            eta = (len(dataloader) - i) * batch_time

            s = "[Val: {:4d}/{:4d}] batch_time: {:.2f}s eta: {:s}".format(
                i, len(dataloader), batch_time, str(timedelta(seconds=int(eta)))
            )

            with open(os.path.join(args.savedir, args.name, "individual_val_output.txt"), "a") as fp:
                fp.write(s + '\n')
            print(s)
            t = time.perf_counter()

            for j in range (0, len(image)):
                # Visualize predictions
                vis_img = transforms.ToPILImage()(image[j].detach().cpu())
                pred_vis = pred_ms[1].detach().cpu()
                label_vis = gt_ms[1]
            
                # Visualize scores
                vis_fig = visualize_score(
                    pred_vis[j],
                    label_vis[j],
                    grid2d[j],
                    vis_img,
                    iou_per_sample[j],
                    iou_dict['s200_iou_per_sample'][j],
                    num_classes, 
                    args,
                )
                plt.savefig(
                    os.path.join(
                        args.savedir,
                        args.name,
                        "val_output_epoch{}_iter{}_iou_{}_from_{}.png".format(epoch, j, args.iou, args.load_ckpt),
                    )
                )
                for k in range (num_classes):
                    iou_mean[k] += iou_dict['s200_iou_per_sample'][j][k]

        iou_mean = iou_mean / len(image)
        total_iou_mean = 0
        for l in range (num_classes):
            print(args.pred_classes_nusc[l], " = ", iou_mean[l])
            total_iou_mean += iou_mean[l]
        print("Overal IOU mean :", total_iou_mean/num_classes)
            

    print("\n==> Validation epoch complete")

    # Calculate per class IoUs over set
    scales = [pred.shape[-1] for pred in pred_ms]

    ms_cumsum_iou_per_class = torch.stack(
        [epoch_iou["s{}_iou_per_class".format(scale)] for scale in scales]
    )
    ms_count_per_class = torch.stack(
        [epoch_iou["s{}_class_count".format(scale)] for scale in scales]
    )

    ms_ious_per_class = (
        (ms_cumsum_iou_per_class / (ms_count_per_class + 1e-6)).cpu().numpy()
    )
    ms_mean_iou = ms_ious_per_class.mean(axis=1)

    # Calculate per class loss over set
    ms_cumsum_loss_per_class = torch.stack(
        [epoch_loss_per_class["s{}_loss_per_class".format(scale)] for scale in scales]
    )
    ms_loss_per_class = (
        (ms_cumsum_loss_per_class / (ms_count_per_class + 1)).cpu().numpy()
    )
    total_loss = ms_loss_per_class.mean(axis=1).sum()

    with open(os.path.join(args.savedir, args.name, "individual_val_loss.txt"), "a") as f:
        f.write("\n")
        f.write(
            "{},".format(epoch)
            + "{},".format(float(total_loss))
            + "".join("{},".format(v) for v in ms_mean_iou)
        )

    with open(os.path.join(args.savedir, args.name, "individual_val_ious.txt"), "a") as f:
        f.write("\n")
        f.write(
            "Epoch: {},\n".format(epoch)
            + "Total Loss: {},\n".format(float(total_loss))
            + "".join(
                "s{}_ious_per_class: {}, \n".format(s, v)
                for s, v in zip(scales, ms_ious_per_class)
            )
            + "".join(
                "s{}_loss_per_class: {}, \n".format(s, v)
                for s, v in zip(scales, ms_loss_per_class)
            )
        )


def parse_args():
    parser = ArgumentParser()

    # ----------------------------- Data options ---------------------------- #
    parser.add_argument(
        "--root",
        type=str,
        default="/Users/quentin/Documents/DLAV/translating-images-into-maps-main/nuscenes_data",
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
    return parser.parse_args()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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

    if args.cuda_available:
       num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 0
    args.num_gpu = num_gpus


def main():
    # Parse command line arguments
    args = parse_args()
    init(args)

    # Create experiment
    print("loading val data")
    val_data = nuScenesMaps(
        root=args.root,
        split=args.val_split,
        grid_size=args.grid_size,
        grid_res=args.grid_res,
        classes=args.load_classes_nusc,
        dataset_size=args.data_size,
        desired_image_size=args.desired_image_size,
        mini=True,
        gt_out_size=(200, 200),
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=src.data.collate_funcs.collate_nusc_s,
        drop_last=True,
        pin_memory=True
    )

    # Build model
    model = networks.__dict__[args.model_name](
        num_classes=len(args.pred_classes_nusc),
        frontend=args.frontend,
        grid_res=args.grid_res,
        pretrained=args.pretrained,
        img_dims=args.desired_image_size,
        z_range=args.z_intervals,
        h_cropped=args.cropped_height,
        dla_norm=args.dla_norm,
        additions_BEVT_linear=args.bevt_linear_additions,
        additions_BEVT_conv=args.bevt_conv_additions,
        dla_l1_n_channels=args.dla_l1_nchannels,
        n_enc_layers=args.n_enc_layers,
        n_dec_layers=args.n_dec_layers,
    )

    if args.pretrained_bem:
        pretrained_pth = os.path.join(pretrained_model_dir, args.load_ckpt)
        pretrained_dict = torch.load(pretrained_pth, map_location=torch.device('cpu'))["model"]
        mod_dict = OrderedDict()
        # # Remove "module" from name
        for k, v in pretrained_dict.items():
            if any(module in k for module in args.ignore):
                continue
            else:
                name = k[7:]
                mod_dict[name] = v

        model.load_state_dict(mod_dict, strict=False)
        print("loaded pretrained model")

    if args.cuda_available:
        device = torch.device("cuda")
    else:
        device = "cpu"
    model = nn.DataParallel(model)
    model.to(device)

    # Setup optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), args.lr, )
    else:
        optimizer = optim.__dict__[args.optimizer](
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)

    # Check if saved model checkpoint exists
    model_dir = os.path.join(args.savedir, args.name)
    checkpt_fn = sorted(
        [
            f
            for f in os.listdir(model_dir)
            if os.path.isfile(os.path.join(model_dir, f)) and args.load_ckpt in f
        ]
    )
    if len(checkpt_fn) != 0:
        model_pth = os.path.join(model_dir, checkpt_fn[-1])
        ckpt = torch.load(model_pth, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["scheduler"])
        epoch_ckpt = ckpt["epoch"] + 1
        print("validating {}".format(checkpt_fn[-1]))
    else:
        epoch_ckpt = 1
        pass

    if args.cuda_available:
        torch.cuda.empty_cache() 
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    validate(args, val_loader, model)


if __name__ == "__main__":
    main()
