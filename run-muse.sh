#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=30192M   # memory per CPU core
#SBATCH -J "MUSE"   # job name
#SBATCH --gpus=1

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source /fslhome/pipoika3/anaconda3/etc/profile.d/conda.sh
conda activate unsup

python /fslhome/pipoika3/MUSE/unsupervised.py --src_lang en --tgt_lang fa --src_emb data/fasttext/Sorenson.en-US.vec --tgt_emb data/fasttext/Sorenson.fa-IR.vec --n_refinement 5
python /fslhome/pipoika3/MUSE/unsupervised.py --src_lang en --tgt_lang fa --src_emb data/fasttext/Sorenson-withOPUS.en-US.vec --tgt_emb data/fasttext/Sorenson-withOPUS.fa-IR.vec --n_refinement 5
