#!/bin/bash

#SBATCH -J dlav-bev-training
#SBATCH --chdir /home/delfosse/venvs/dlav/dlav
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --qos dlav
#SBATCH --account civil-459-2023
#SBATCH -o slurm.%j.out # STDOUT
#SBATCH -e slurm.%j.err # STDERR
#SBATCH --time 24:00:00

#Load any modules that your program needs
module load gcc/8.4.0-cuda python/3.7.7 cuda/11.6.2
source /home/delfosse/venvs/dlav/bin/activate

nvcc --version
nvidia-smi

#Run your program
python train.py --name "27_04_23_11_08" --savedir="/work/scitas-share/datasets/Vita/civil-459/Nuscenes_bev/pretrained_models" --val-interval 1 --root="/work/scitas-share/datasets/Vita/civil-459/Nuscenes_bev" --nusc-version="v1.0-trainval" --train-split="train_roddick" --val-split="val_roddick" --data-size=1 --epochs=40 --batch-size=8 --accumulation-steps=1 --cuda-available=1 --iou=0