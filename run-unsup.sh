#!/bin/bash

#SBATCH --time=3-00:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=60192M   # memory per CPU core
#SBATCH -J "trans"   # job name
#SBATCH -C 'pascal'   # features syntax (use quotes): -C 'a&b&c&d'
#SBATCH --gpus=1

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source /fslhome/pipoika3/anaconda3/etc/profile.d/conda.sh
conda activate unsup

export JOB_ID=$SLURM_JOB_ID
export BASE_NAME="Sorenson-5000"
export SEED_ID="none"
export SEED_BASE="none"
python3 tokenizer.py $BASE_NAME
python3 unsup.py $BASE_NAME $SEED_ID $SEED_BASE

