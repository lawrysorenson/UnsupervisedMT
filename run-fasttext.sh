#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=30192M   # memory per CPU core
#SBATCH -J "fasttext"   # job name

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

/fslhome/pipoika3/fastText/fasttext skipgram -input data/split/Sorenson-train-token.en-US -output data/fasttext/Sorenson.en-US -dim 300 -epoch 20
/fslhome/pipoika3/fastText/fasttext skipgram -input data/split/Sorenson-train-token.fa-IR -output data/fasttext/Sorenson.fa-IR -dim 300 -epoch 20
/fslhome/pipoika3/fastText/fasttext skipgram -input data/split/OPUS-train-token.en-US -output data/fasttext/OPUS.en-US -dim 300 -epoch 20
/fslhome/pipoika3/fastText/fasttext skipgram -input data/split/OPUS-train-token.fa-IR -output data/fasttext/OPUS.fa-IR -dim 300 -epoch 20
