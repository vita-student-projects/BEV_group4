# From Monocular Camera Image to BEV - Improving Pedestrians Detection
#### Quentin Delfosse, Gloria Mellinand
<div>
<img src="images/image_to_bev_motivation.gif"></img>
</div>
<br />

This code was built upon a pre-existing [Image to BEV deep learning model](https://github.com/avishkarsaha/translating-images-into-maps/), based on the paper [Translating Images Into Maps](https://arxiv.org/abs/2110.00966)

## Environment Setup

Clone this repo: `git clone git@github.com:deepakduggirala/translating-images-into-maps.git`

### Local environment

Create a conda environment with python 3 and install dependencies

```bash
conda create -n tim python=3
conda activate tim

conda install -c pytorch pytorch torchvision
conda install -c conda-forge pytorch-lightning torchinfo
conda install -c matplotlib jupyterlab
conda install numpy pyquaternion shapely pandas
conda install opencv
pip install lmdb nuscenes-devkit
```

### Carbonate Deep Learning Cluster

Get access to Carbonate deep learning cluster and log into a node.

```bash
module load deeplearning/2.10.1
pip install pyquaternion shapely lmdb opencv-python nuscenes-devkit
```

## Data

Download the nuScenes dataset after creating an account from https://www.nuscenes.org/nuscenes. Specifically download `v1.0-trainval` (training and validation dataset consisting of 850 scenes)

Extract the metadata files into a folder (root).

Extract only the CAM_FRONT data from the downloaded tar zip files into the root folder.
```bash
tar -xvf v1.0-trainval01_blobs.tgz samples/CAM_FRONT
tar -xvf v1.0-trainval02_blobs.tgz samples/CAM_FRONT
tar -xvf v1.0-trainval03_blobs.tgz samples/CAM_FRONT
tar -xvf v1.0-trainval04_blobs.tgz samples/CAM_FRONT
tar -xvf v1.0-trainval05_blobs.tgz samples/CAM_FRONT
tar -xvf v1.0-trainval06_blobs.tgz samples/CAM_FRONT
tar -xvf v1.0-trainval07_blobs.tgz samples/CAM_FRONT
tar -xvf v1.0-trainval08_blobs.tgz samples/CAM_FRONT
tar -xvf v1.0-trainval09_blobs.tgz samples/CAM_FRONT
tar -xvf v1.0-trainval10_blobs.tgz samples/CAM_FRONT
```

### Ground truths

This contains the ground truth maps which have already been generated for
the mini dataset, the input images and intrinsics.

- create a directory called `lmdb` inside root directory
- Download the directory `semantic_maps_new_200x200` from `https://drive.google.com/drive/folders/1-NnCOQqk-nmX82myvYkbKJXn73sFZ9wl` and place it in the root folder.

### Splits
Split files are text files that contain the sample_tokens to be used for training and validation.

There are 850 scenes in v1.0-trainval datatset. 698 scenes used for training and 148 scenes are used for validations.  

However 4 scenes are neither in training nor val datasets: scene-0410, scene-0641, scene-0805, scene-0910


- Download `splits` from `https://drive.google.com/drive/folders/10_XETGfPiCs1pcWc_fVXjvH-blh_HapQ` and place it in the root folder



## Training
Submit a job after changing appropriate command line args to `train.py` in the script `job.script.sh`

```bash
sbatch job.script.sh
```

Notable options:
- `--name`: Name of the experiment. All the checkpoints and results are stored in `experiments/<name>`
- `--root`: Nuscenes data location
- `--train-split`: The list of sample tokens to train on
- `--val-split`: The list of sample tokens to validate on
- `--epochs`: Train till this epoch. If the training has to resumed after 20 epochs and trained till 30 epochs, set this value to 30.
- `--batch-size`: Largest batch size that can work in Carbonate is 32. But the batch size used for training is 8.

## Validation / Inference
```bash
sbatch job.val.sh # validation
sbatch job.evaluate.sh # inference
```

## Fixes to make the original code work
Load images from nuScenes dataset insteead of lmdb (src/data/dataloader.py - line 140-146)

```python
original_nusenes_dir = Path.resolve(Path('../v1.0-mini/samples/CAM_FRONT'))
new_cam_path = original_nusenes_dir / Path(cam_path).name
image = Image.open(new_cam_path).convert(mode='RGB')
```

Construct pickled key with pickle version 3. (src/data/dataloader.py - line 140, 154)

```python
gtmaps_key = [pickle.dumps("{}___{}".format(id, cls), 3) for cls in classes]
```

Change these trianing parameters
- `--val-interval 1`
- `--data-size=1`
- `--accumulation-steps=1`
- `--lr=5e-5`
