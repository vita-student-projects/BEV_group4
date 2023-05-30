#!/bin/bash

#SBATCH -J job_name
#SBATCH -p gpu
#SBATCH -A r00068
#SBATCH -o filename_%j.txt
#SBATCH -e filename_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node v100:1
#SBATCH --time=2:00:00
#SBATCH --mem=16gb

#Load any modules that your program needs
module load deeplearning/2.9.1

#Run your program
python src/data/image_db.py 4