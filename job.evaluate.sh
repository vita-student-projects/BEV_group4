#!/bin/bash

#SBATCH -J val_job
#SBATCH -p gpu
#SBATCH -A r00068
#SBATCH -o job_logs/filename_%j.txt
#SBATCH -e job_logs/filename_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=deduggi@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node v100:4
#SBATCH --time=07:00:00
#SBATCH --mem=55gb

#Load any modules that your program needs
module load deeplearning/2.10.0

#Run your program
python inference.py --name "tiim_28k" --root="/N/slate/deduggi/nuScenes-trainval" --val-split="evaluation" --data-size=1 --batch-size=8 --accumulation-steps=1
