#!/bin/bash

#SBATCH --time=5:30:00   # walltime
#SBATCH --ntasks-per-node=1 # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=100G   # memory per CPU core
#SBATCH --gpus=1
#SBATCH --qos=dw87
#SBATCH --partition=dw
##SBATCH --qos=cs
##SBATCH --partition=cs
##SBATCH -J "test"   # job name
##SBATCH --output=%x_%j.out

nvidia-smi
conda activate cs673

srun python3 experiment.py --out_dir run_0


