import os
import sys
from pathlib import Path
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
import torchvision.transforms as transforms
import matplotlib as mpl
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader

import src
from src import utils
import src.data.collate_funcs
import src.model.network as networks
from src.data.dataloader import nuScenesMaps
from src.utils import MetricDict

from src.data.dataloader import nuScenesMaps, read_split
from argslib import parse_args, init

import cv2
from PIL import Image


def load_checkpoint(args, model, ckpt_name=None, ckpt_epoch=None):
    model_dir = Path(args.savedir) / args.name
    if not ckpt_name:
        if ckpt_epoch:
            ckpt_name = f'checkpoint-{str(ckpt_epoch).zfill(4)}.pth.gz'
        else:
            checkpt_fn = sorted(
                [
                    f
                    for f in os.listdir(str(model_dir))
                    if os.path.isfile(os.path.join(model_dir, f)) and ".pth.gz" in f
                ]
            )
            ckpt_name = checkpt_fn[-1]
    ckpt_fname = model_dir / ckpt_name
    print(f'Loading checkpoint {ckpt_name}')
    assert (ckpt_fname.exists())

    model_pth = os.path.join(model_dir, str(ckpt_fname))
    if torch.cuda.is_available():
        ckpt = torch.load(model_pth)
    else:
        ckpt = torch.load(model_pth, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt["model"])


def get_model(args):
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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = nn.DataParallel(model)
    _ = model.to(device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    return model


def time_val_epoch(N, fname=None):
    i = 0

    # Print summary
    while True:
        start_time = time.perf_counter()
        yield
        i = i + 1
        batch_time = (time.perf_counter() - start_time)
        eta = (N - i) * batch_time

        s = "[Val: {:4d}/{:4d}] batch_time: {:.2f}s eta: {:s}".format(
            i, N, batch_time, str(timedelta(seconds=int(eta)))
        )

        if fname:
            with open(fname, "a") as fp:
                fp.write(s + '\n')
        print(s)



def evaluate(model, dataloader, video_path, args):
    model.eval()
    res_100 = None
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("frame_count :", frame_count)
    time_epoch_gen = time_val_epoch(N=len(dataloader))
    next(time_epoch_gen)

    for i, ((image_mini, calib, grid2d), (cls_map, vis_mask)) in enumerate(dataloader):
        calib_copy = calib
        print("calib_copy :", calib_copy)
        grid2d_copy = grid2d

    for i in range(frame_count):
        success, frame = cap.read()
        if not success:
            break
        
        # cropping to remove car dashboard in test files
        startX, startY, endX, endY = 0, 0, 1920, 600
        # frame = frame[startY:endY, startX:endX]
        # frame[endY:, :] = [0, 0, 0]
        print("height, width, _ :", frame.shape)

        image_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        image = Image.fromarray(frame)
        image = image_calib(image, calib_copy)
        
        # Pil Image to Numpy array
        image = np.asarray(image)
        image = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(0) #.float()

        image_np = image.squeeze().numpy()

        # # Transpose the dimensions to (height, width, channels)
        # image_np = np.transpose(image_np, (1, 2, 0))

        # # Display the image using Matplotlib
        # plt.imshow(image_np)
        # plt.axis('off')
        # plt.show()

        with torch.no_grad():
            # print("IMAGE :", image.shape)

            pred_ms = model(image, calib_copy, grid2d_copy)

            # Upsample the largest prediction to 200x200
            pred_200x200 = F.interpolate(
                pred_ms[0], size=(200, 200), mode="bilinear"
            )
            pred_ms = [pred_200x200, *pred_ms]
            pred_ms_cpu = [pred.detach().cpu() for pred in pred_ms]

            if res_100 is not None:
                res_100 = torch.cat((res_100, pred_ms_cpu[1]))
            else:
                res_100 = torch.cat((pred_ms_cpu[1],))

            _, predicted_classes = torch.max(pred_ms_cpu[1], dim=1)
            # print("Class predictions per pixel :", predicted_classes)

            display_predictions_2(pred_ms, image_rgb, args)
            # display_predictions_with_classes(image_rgb, predicted_classes[0], args)
            # display_predictions_with_input(image_rgb, predicted_classes[0], args)
            # display_predictions(image_rgb, predicted_classes[0], args)
            
            # Create new directory for images
            directory = os.path.join(args.savedir, args.name,args.video_name)
            os.makedirs(directory, exist_ok=True)

            plt.savefig(
                os.path.join(
                    directory,
                    "inference_2_output_image_{}_2.png".format(i),
                )
            )

        next(time_epoch_gen)

    return res_100

def image_calib(image, calib):

    og_w, og_h = image.size
    print("image.size :", image.size)
    desired_w, desired_h = (1280, 720)
    scale_w, scale_h = desired_w / og_w, desired_h / og_h
    # Scale image
    image = image.resize((int(image.size[0] * scale_w), int(image.size[1] * scale_h)))
    # Pad images to the same dimensions
    w = image.size[0]
    h = image.size[1]
    delta_w = desired_w - w
    delta_h = desired_h - h
    pad_left = int(delta_w / 2)
    pad_right = delta_w - pad_left
    pad_top = int(delta_h / 2)
    pad_bottom = delta_h - pad_top
    left = 0 - pad_left
    right = pad_right + w
    top = 0 - pad_top
    bottom = pad_bottom + h
    image = image.crop((left, top, right, bottom))

    return image


# Updated function to display the predited bev with input image and color codes
def display_predictions_2(pred_ms, image_rgb, args):
    scores = pred_ms[1].detach().cpu()
    score = scores[0]
    class_idx = torch.arange(len(score)) + 1
    logits = score.clone().cpu() * class_idx.view(-1, 1, 1)
    logits, _ = logits.max(dim=0)
    
    score = (score.detach().clone().cpu()>0.5).float() * class_idx.view(-1, 1, 1)

    cls_idx = score.clone()
    cls_idx = cls_idx.argmax(dim=0)
    cls_idx = cls_idx.numpy()

    num_classes = len(args.pred_classes_nusc)
    
    np.set_printoptions(threshold=sys.maxsize)
    # print("cls_idx :", cls_idx)

    color_codes = cv2.applyColorMap(np.uint8(cls_idx * (255/num_classes)), cv2.COLORMAP_JET)
    color_codes = cv2.cvtColor(color_codes, cv2.COLOR_BGR2RGB)
    # print("color_codes :", color_codes)

    score, _ = score.max(dim=0) 

    # width, height = image_rgb.size
    # print("width :", width)
    # print("height :", height)
    # dpi = 72
    # fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig = plt.figure(num="score", figsize=(10, 8))
    fig.clear()
    gs = mpl.gridspec.GridSpec(1, 3, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1]) 
    ax2 = fig.add_subplot(gs[0, 2])

    ax0.imshow(image_rgb)
    ax0.axis('off')

    ax1.imshow(color_codes, origin='lower') 
    ax1.axis('off')
    
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=0, vmax=num_classes-1)
    class_names = [args.pred_classes_nusc[i] for i in range(num_classes)]
    handles = [mpl.patches.Patch(color=cmap(norm(i)), label=f"{class_names[i]}") for i in range(num_classes)]
    # for i in range (num_classes):
    #     print("cmap(norm(i) :", cmap(norm(i)))
    ax2.legend(handles=handles, loc='center', bbox_to_anchor=(0.5, 0.5))
    # ax2.imshow(color_codes, origin='lower') 
    ax2.axis('off')

# Original function to display the predited bev
def display_predictions(image, predicted_classes, args):
    width, height = image.size
    print("width :", width)
    print("height :", height)
    dpi = 72

    num_classes = len(args.pred_classes_nusc)

    color_codes = cv2.applyColorMap(np.uint8(predicted_classes * (255/num_classes)), cv2.COLORMAP_JET)
    color_codes = cv2.cvtColor(color_codes, cv2.COLOR_BGR2RGB)

    class_names = [args.pred_classes_nusc[i] for i in range(num_classes)]

    # Plot the original image and the output image
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    plt.imshow(color_codes)
    # plt.set_title("Predicted Classes")
    plt.axis("off")

    # plt.show()

    return fig


# Original function to display the predited bev with input image
def display_predictions_with_input(image, predicted_classes, args):
    num_classes = len(args.pred_classes_nusc)

    color_codes = cv2.applyColorMap(np.uint8(predicted_classes * (255/num_classes)), cv2.COLORMAP_JET)
    color_codes = cv2.cvtColor(color_codes, cv2.COLOR_BGR2RGB)

    class_names = [args.pred_classes_nusc[i] for i in range(num_classes)]

    # Plot the original image and the output image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2.imshow(color_codes)
    ax2.set_title("Predicted Classes")
    ax2.axis("off")

    # plt.show()

    return fig

# Original function to display the predited bev with image and color codes
def display_predictions_with_classes(image, predicted_classes, args):
    num_classes = len(args.pred_classes_nusc)
    class_names = [args.pred_classes_nusc[i] for i in range(num_classes)]

    color_codes = cv2.applyColorMap(np.uint8(predicted_classes * (255/num_classes)), cv2.COLORMAP_JET)
    color_codes = cv2.cvtColor(color_codes, cv2.COLOR_BGR2RGB)

    # Create a legend with class names and colors
    legend_img = np.zeros((num_classes * 30, 200, 3), dtype=np.uint8)
    for i in range(num_classes):
        legend_img[i * 30 : (i + 1) * 30, :] = color_codes[0, 0, :] if i in predicted_classes else [0, 0, 0]
        cv2.putText(
            legend_img,
            f"{class_names[i]} ({i})",
            (10, i * 30 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Plot the original image, the output image, and the legend
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2.imshow(color_codes)
    ax2.set_title("Predicted Classes")
    ax2.axis("off")
    ax2.set_ylim(ax2.get_ylim()[::-1])
    ax2.set_xlim(ax2.get_xlim()[::-1])

    ax3.imshow(legend_img)
    ax3.set_title("Legend")
    ax3.axis("off")

    # plt.show()

    return fig 


def main():
    args = parse_args(notebook=False)
    init(args)

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
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=src.data.collate_funcs.collate_nusc_s,
        drop_last=True,
        pin_memory=True
    )

    video_name = args.video_name # "test1"
    video_path = args.video_root # "/Users/quentin/Downloads"
    video_file = os.path.join(video_path, video_name + ".mp4")

    model = get_model(args)
    # load_checkpoint(args, model, ckpt_epoch=ckpt_epoch)
    load_checkpoint(args, model, ckpt_name=args.load_ckpt)
    # res_100 = evaluate(model, val_loader)
    res_100 = evaluate(model, val_loader, video_file, args)

    # print("res_100 :", res_100)

    # experiment_dir = Path(args.savedir) / args.name
    # results_dir = experiment_dir / 'inference_results'
    # results_dir.mkdir(exist_ok=True)
    torch.save(res_100, os.path.join(args.savedir, args.name, args.video_name))


if __name__=='__main__':
    main()