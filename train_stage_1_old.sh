#!/bin/bash
#SBATCH --job-name=m1
#SBATCH --account=Project_2002932
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --mem=150G
#SBATCH --cpus-per-task=10
#SBATCH --time=1-20:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=END
# #SBATCH --mail-user=joaquin.j.rives@utu.fi

# #SBATCH -o output_%j.txt   <------- TODO
# #SBATCH -e error_%j.txt

# *******************************************************************
# => USEFUL COMMANDS:
# scontrol show job <id> (to see all info)
# squeue (-p) <partition> (to see the queue)
# squeue -u rivesgam
# scancel <jobid>
# seff <jobid> (info about the requested resources)
# watch -n1 cat *45345.out (to automatically cat the output every n seconds)
# *******************************************************************

# example run commands

module load tensorflow/1.14.0

srun python3 train_stage_1.py

# This script will print some usage statistics to the
# end of the standard out file
# Use that to improve your resource request estimate
# on later jobs.
seff $SLURM_JOBID
