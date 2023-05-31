# From Monocular Camera Image to BEV - Improving Pedestrians Detection
#### Quentin Delfosse, Gloria Mellinand

This code was built upon a pre-existing [Image to BEV deep learning model](https://github.com/avishkarsaha/translating-images-into-maps/), based on the paper [Translating Images Into Maps](https://arxiv.org/abs/2110.00966). 
This code was written using python 3.7. and was trained on the nuScenes dataset.
Please refer to the repository's Read Me for dependencies and datasets to install.

## Using the code
The first step is to create a folder named "translating-images-into-maps-main" and download all files into it.
Then, due to large file size, the latest checkpoint of our training and the mini nuScenes dataset used for validation can be downloaded [from this Google Drive](https://drive.google.com/drive/folders/0ALp6UvHAP1hAUk9PVA). These folders should be added directly in the "translating-images-into-maps-main" directory. 

Below is the list of the required libraries for this repo:
```pytorch
opencv
numpy
pyquaternion
shapely
lmdb
nuscenes-devkit
pillow
matplotlib
torchvision
descartes
scipy
tensorboard
scikit-image
cv2
```

## Project Context
This project was made in the context of the Deep Learning for Autonomous Vehicules course CIVIL-459, taught by Professor Alexandre Alahi at EPFL. We were supervised by doctoral student Yuejiang Liu. 
The main goal of the course's project is to develop a deep learning model that can be used onboard a Tesla autopilot system. As for our group, we have been looking into the transformation from monocular camera images to bird's eye view. This can be done by using semantic segmentation to classify elements such as cars, sidewalk, pedestrians and the horizon. 

During our research on Monocular images to BEV deep learning models, we have noticed that information concerning pedestrians was lost during segmentation, resulting in poor classification. More information about our research can be found on the [Drive](https://drive.google.com/drive/folders/0ALp6UvHAP1hAUk9PVA). As seen on the image below, 

<div>
<img src="1_evaluation.png"></img>
</div>
<br />


### Contribution





## References
<a id="1">[1]</a> 
Saha, Avishkar and Mendez, Oscar and Russell, Chris and Bowden, Richard (2022). 
Translating Images into Maps.
2022 IEEE International Conference on Rbotics and Automation (ICRA)

<a id="1">[1]</a> 
Saha, Avishkar and Mendez, Oscar and Russell, Chris and Bowden, Richard (2021). 
Enabling spatio-temporal aggregation in birds-eye-view vehicle estimation.
2021 IEEE International Conference on Robotics and Automation (ICRA) (pages 5133-5139)
