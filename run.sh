#!/bin/bash

#SBATCH --job-name=project
#SBATCH --account=cse595s001f25_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16g
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=project.out

python3 main.py