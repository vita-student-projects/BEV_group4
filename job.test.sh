#!/bin/bash

#SBATCH -J job_name
#SBATCH -p gpu
#SBATCH -A r00068
#SBATCH -o job_logs/filename_%j.txt
#SBATCH -e job_logs/filename_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node v100:4
#SBATCH --time=0:05:00

#Load any modules that your program needs
module load deeplearning/2.9.1

#Run your program
python test.py