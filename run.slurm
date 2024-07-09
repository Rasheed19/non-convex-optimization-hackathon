#!/bin/bash
#SBATCH --job-name=team4A
#SBATCH --account=ic084
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5

cd $SLURM_SUBMIT_DIR

export OMP_NUM_THREADS=1

srun python run.py --device cuda
