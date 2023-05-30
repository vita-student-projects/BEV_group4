#!/bin/bash

#SBATCH -J job_name
#SBATCH -p gpu
#SBATCH -A r00068
#SBATCH -o job_logs/filename_%j.txt
#SBATCH -e job_logs/filename_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=deduggi@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node v100:4
#SBATCH --time=47:59:00
#SBATCH --mem=55gb

#Load any modules that your program needs
module load deeplearning/2.10.0

#Run your program
python train.py --name "tiim_28k" --val-interval 1 --root="/N/slate/deduggi/nuScenes-trainval" --train-split="train_roddick" --val-split="val_roddick" --data-size=1 --epochs=40 --batch-size=8 --accumulation-steps=1
