#!/bin/bash
#SBATCH --job-name=team4A
#SBATCH --account=ic084
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5

module load python/3.11.5-gpu

cd $SLURM_SUBMIT_DIR
extend-venv-activate ../venv
source ../venv/bin/activate

export OMP_NUM_THREADS=1

srun python run.py --device cuda
