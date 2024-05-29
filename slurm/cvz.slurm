#!/bin/bash
## Job Name
#SBATCH --job-name=flair-cvz
#SBATCH --account=astro
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=0:30:00
#SBATCH --mem=10G
#SBATCH -o /gscratch/dirac/flair/slurm/logs/cvz_%a_%A.out
#SBATCH -e /gscratch/dirac/flair/slurm/logs/cvz_%a_%A.err
#SBATCH --chdir=/gscratch/dirac/flair/pipeline
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tomwagg@uw.edu          ## <--- CHANGE THIS TO YOUR EMAIL!!!
#SBATCH --export=all

source ~/.bashrc

conda activate /gscratch/dirac/flair/conda/envs/flair
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gscratch/dirac/flair/conda/lib

LK_CACHE_PATH=/gscratch/scrubbed/flair/
SCRIPT_PATH=/gscratch/dirac/flair/pipeline/cvz.py
OUTPUT_PATH=/gscratch/dirac/flair/output/

python $SCRIPT_PATH -o $OUTPUT_PATH -l $LK_CACHE_PATH -t tic272272592 -i 0
