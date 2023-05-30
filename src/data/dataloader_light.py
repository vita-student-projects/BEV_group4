import io
import os
import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from src import utils


class nuScenesMaps(Dataset):

    def __init__(
            self,
            root="temp",
            split="train_mini",
            grid_size=(50.0, 50.0),
            grid_res=1.0,
            classes=None,
            dataset_size=1.0,
            mini=False,
            desired_image_size=(1280, 720),
            gt_out_size=(100, 100),
            nusc_data_file="eval_data.pickle"
    ):
        if classes is None:
            self.classes = [
                "bus",
                "bicycle",
                "car",
                "construction_vehicle",
                "motorcycle",
                "trailer",
                "truck",
                "pedestrian",
            ]
        self.dataset_size = dataset_size
        self.desired_image_size = desired_image_size
        self.gt_out_size = gt_out_size

        # paths for data files
        self.root = os.path.join(root)
        self.gtmaps_db_path = os.path.join(
            root, "lmdb",
            "semantic_maps_new_200x200"
        )

        self.tokens = read_split(
            os.path.join(root, "splits", "{}.txt".format(split))
        )
        self.gtmaps_db = lmdb.open(
            path=self.gtmaps_db_path,
            readonly=True,
            readahead=False,
            max_spare_txns=128,
            lock=False,
        )

        # Set classes
        self.classes = list(classes)
        self.classes.append("lidar_ray_mask_dense")
        self.class2idx = {
            name: idx for idx, name in enumerate(self.classes)
        }
        self.nusc_classes = [
            "vehicle.bus",
            "vehicle.bicycle",
            "vehicle.car",
            "vehicle.construction",
            "vehicle.motorcycle",
            "vehicle.trailer",
            "vehicle.truck",
            "human.pedestrian",
        ]
        self.nuscclass2idx = {
            name: idx for idx, name in enumerate(self.nusc_classes)
        }
        # load FOV mask
        self.fov_mask = Image.open(
            os.path.join(root, "lmdb", "semantic_maps_new_200x200", "fov_mask.png")
        )
        # Make grid
        self.grid2d = utils.make_grid2d(grid_size, (-grid_size[0] / 2.0, 0.0), grid_res)
        self.cam_front_dir = Path.resolve(Path(root) / 'samples/CAM_FRONT')
        with open(os.path.join(self.root, nusc_data_file), 'rb') as f:
            self.nusc_data = pickle.load(f)

    def __len__(self):
        return int(len(self.tokens) * self.dataset_size - 1)

    def __getitem__(self, index):
        # Load sample ID
        sample_token = self.tokens[index]
        cam_path = self.nusc_data[sample_token]['filename']
        cam_id = Path(cam_path).stem

        # Load intrinsincs
        calib = self.nusc_data[sample_token]['calib']
        calib = np.array(calib)

        # Load input images
        # image_input_key = pickle.dumps(cam_id, 3)
        # image_input_key = cam_id.encode('utf-8')
        # with self.images_db.begin() as txn:
        #     value = txn.get(key=image_input_key)
        #     image = Image.open(io.BytesIO(value)).convert(mode='RGB')

        new_cam_path = self.cam_front_dir / Path(cam_path).name
        image = Image.open(new_cam_path).convert(mode='RGB')

        # resize/augment images
        image, calib = self.image_calib_pad_and_crop(image, calib)
        image = to_tensor(image)
        calib = to_tensor(calib).reshape(3, 3)

        # Load ground truth maps
        gtmaps_key = [pickle.dumps("{}___{}".format(cam_id, cls), 3) for cls in self.classes]
        with self.gtmaps_db.begin() as txn:
            value = [txn.get(key=key) for key in gtmaps_key]
            gtmaps = [Image.open(io.BytesIO(im)) for im in value]

        # each map is of shape [1, 200, 200]
        mapsdict = {cls: to_tensor(_map) for cls, _map in zip(self.classes, gtmaps)}
        mapsdict["fov_mask"] = to_tensor(self.fov_mask)
        mapsdict = self.merge_map_classes(mapsdict)

        # Create visbility mask from lidar and fov masks
        lidar_ray_mask = mapsdict['lidar_ray_mask_dense']
        fov_mask = mapsdict['fov_mask']
        vis_mask = lidar_ray_mask * fov_mask
        mapsdict['vis_mask'] = vis_mask

        del mapsdict['lidar_ray_mask_dense'], mapsdict['fov_mask']

        # downsample maps to required output resolution
        mapsdict = {
            cls: F.interpolate(cls_map.unsqueeze(0), size=self.gt_out_size).squeeze(0)
            for cls, cls_map in mapsdict.items()
        }

        # apply vis mask to maps
        mapsdict = {
            cls: cls_map * mapsdict['vis_mask'] for cls, cls_map in mapsdict.items()
        }

        cls_maps = torch.cat(
            [cls_map for cls, cls_map in mapsdict.items() if 'mask' not in cls], dim=0
        )
        vis_mask = mapsdict['vis_mask']

        return (
            image, cls_maps, vis_mask, calib, self.grid2d
        )

    def merge_map_classes(self, mapsdict):
        classes_to_merge = ["drivable_area", "road_segment", "lane"]
        merged_class = 'drivable_area'
        maps2merge = torch.stack([mapsdict[k] for k in classes_to_merge])  # [n, 1, 200, 200]
        maps2merge = maps2merge.sum(dim=0)
        maps2merge = (maps2merge > 0).float()
        mapsdict[merged_class] = maps2merge
        del mapsdict['road_segment'], mapsdict['lane']
        return mapsdict

    def image_calib_pad_and_crop(self, image, calib):

        og_w, og_h = 1600, 900
        desired_w, desired_h = self.desired_image_size
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

        # Modify calibration matrices
        # Scale first two rows of calibration matrix
        calib[:2, :] *= scale_w
        # cx' = cx - du
        calib[0, 2] = calib[0, 2] + pad_left
        # cy' = cy - dv
        calib[1, 2] = calib[1, 2] + pad_top

        return image, calib


def read_split(filename):
    """
    Read a list of NuScenes sample tokens
    """
    with open(filename, "r") as f:
        lines = f.read().split("\n")
        return [val for val in lines if val != ""]