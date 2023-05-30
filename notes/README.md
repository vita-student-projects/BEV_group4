# Translating Images to Maps
20221116202713

### Environment setup

Source: https://github.com/avishkarsaha/translating-images-into-maps

```bash
conda create -n tim python=3
conda activate tim

conda install -c pytorch pytorch torchvision
conda install -c conda-forge pytorch-lightning torchinfo
conda install -c matplotlib jupyterlab
conda install numpy pyquaternion shapely
conda install opencv
pip install lmdb nuscenes-devkit

pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable execute_time/ExecuteTime
```

```bash
module load deeplearning/2.9.1
pip install pyquaternion shapely lmdb opencv-python nuscenes-devkit
```

### transfer files and setup repository
```bash
cd autonomous-robotics
git clone git@github.com:deepakduggirala/translating-images-into-maps.git
```


```bash
scp nuscenes_data-20221117T012449Z-001.zip pretrained_models-20221117T012532Z-001.zip deduggi@carbonate.uits.iu.edu:~/autonomous-robotics/
scp v1.0-mini.tgz deduggi@bigred200.uits.iu.edu:~/autonomous-robotics

# In remote machine
mkdir v1.0-mini
tar -xf v1.0-mini.tgz -C v1.0-mini
unzip nuscenes_data-20221117T012449Z-001.zip
mv nuscenes_data translating-images-into-maps/
cp v1.0-mini/v1.0-mini/*.json translating-images-into-maps/nuscenes_data/v1.0-mini/
```

create job.script.sh - carbonate
```bash
#!/bin/bash

#SBATCH -J job_name
#SBATCH -p gpu
#SBATCH -A r00068
#SBATCH -o filename_%j.txt
#SBATCH -e filename_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=deduggi@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node v100:1
#SBATCH --time=10:00:00

#Load any modules that your program needs
module load deeplearning/2.9.1

#Run your program
python train.py
```

create job.script.sh - carbonate
```bash
#!/bin/bash

#SBATCH -J train_mini_job
#SBATCH -p gpu
#SBATCH -o filename_%j.txt
#SBATCH -e filename_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=deduggi@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node 1
#SBATCH --time=10:00:00

#Load any modules that your program needs
module load deeplearning/2.9.1

#Run your program
python train.py
```



Conda environmnet file of other user who got the code running: https://github.com/avishkarsaha/translating-images-into-maps/issues/9


- Download datasets:     https://drive.google.com/drive/folders/1-1dZXeHnPiuqX-w8ruJHqfxBuMYMONRT?usp=sharing
- Download pretrained model:  https://drive.google.com/drive/folders/13-WobzEg9MGcT-tceXE3kwgEv8Q-kUwU?usp=sharing

https://github.com/avishkarsaha/translating-images-into-maps/issues/3 

```
To train on the bigger dataset the procedure is exactly the same and uses the same dataloader. However, the ground truth maps will have to be generated first. I'll provide details on this soon, but in the meantime follow the ground truth generation procedure here: https://github.com/tom-roddick/mono-semantic-maps. You can also then use their dataloader.
```

### Fixes to make code work
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


### Ground Truths
Ground truths are supplied in `translating-images-into-maps/nuscenes_data/lmdb/semantic_maps_new_200x200/` lmdb.

```python
gtmaps_db_path = str(Path.resolve(
    Path(
        './translating-images-into-maps/nuscenes_data/lmdb/semantic_maps_new_200x200/'
    )
))

gtmaps_db = lmdb.open(
    path=gtmaps_db_path,
    readonly=True,
    readahead=False,
    max_spare_txns=128,
    lock=False,
)

# read all keys
with gtmaps_db.begin() as txn:
    gt_keys = list(txn.cursor().iternext(values=False))
    
    
# read ground truths
id = Path(cam_path).stem # 'n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460'
gtmaps_key = [pickle.dumps("{}___{}".format(id, cls), 3) for cls in classes]
with gtmaps_db.begin() as txn:
    value = [txn.get(key=key) for key in gtmaps_key]
    gtmaps = [Image.open(io.BytesIO(im)) for im in value]
```

The ground truths for all 850 scenes (34149 keyframes) are available in `gtmaps_db`

### Splits
Split files are text files that contain the sample_tokens to be used for training and validation.

v1.0-mini:
```
translating-images-into-maps/nuscenes_data/splits/train_mini.txt
translating-images-into-maps/nuscenes_data/splits/val_mini.txt
```

v1.0-trainval:

There are 850 scenes in v1.0-trainval datatset. 698 scenes used for training and 148 scenes are used for validations.  

However 4 scenes are neither in training nor val datasets: scene-0410, scene-0641, scene-0805, scene-0910

```
translating-images-into-maps/nuscenes_data/splits/train_roddick.txt
translating-images-into-maps/nuscenes_data/splits/val_roddick.txt
```