#!/bin/bash

#SBATCH -J dlav-bev-validation
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
#SBATCH --time 01:00:00

#Load any modules that your program needs
module load gcc/8.4.0-cuda python/3.7.7 cuda/11.6.2
source /home/delfosse/venvs/dlav/bin/activate

#Run your program
python validation.py --load-ckpt="checkpoint-epfl-epoch-0016-mini-False-iou-1.pth.gz" --iou=1 --root="/Users/quentin/Documents/DLAV/translating-images-into-maps-main/nuscenes_data"