#!/bin/bash
#SBATCH --job-name=known
#SBATCH --account=Project_2002932
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00

# example run commands

module load tensorflow/1.14.0

srun python3 train_stage_2.py

seff $SLURM_JOBID
