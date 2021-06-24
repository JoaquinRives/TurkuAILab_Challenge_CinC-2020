#!/bin/bash
#SBATCH --job-name=002
#SBATCH --account=Project_2002932
#SBATCH -o array_job_out_%A_%a.out
#SBATCH -e array_job_err_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --array=1-4
#SBATCH --time=2-20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:4

# *******************************************************************
# => USEFUL COMMANDS:
# scontrol show job <id> (to see all info)
# queue (-p) <partition> (to see the queue)
# squeue -u rivesgam
# scancel <jobid>
# seff <jobid> (info about the requested resources)
# watch -n1 cat *45345.out (to automatically cat the output every n seconds)
# *******************************************************************

module load tensorflow/1.14.0 gcc/8.3.0 cuda/10.1.168

# arguments for the python script
list_args="model_1_1 model_1_2 model_1_3 model_1_4"
arr=($list_args)

# SLURM_ARRAY_TASK_ID will be the index/number of the job to run (0, 1, 2...)
srun python train_stage_1.py ${arr[${SLURM_ARRAY_TASK_ID} - 1]}


## commands to manage the batch script
##   submission command
##     sbatch [script-file]
##   status command
##     squeue -u rivesgam
##   termination command
##     scancel [jobid]

#
## This script will print some usage statistics to the
## end of the standard out file
## Use that to improve your resource request estimate
## on later jobs.
#seff $SLURM_JOBID