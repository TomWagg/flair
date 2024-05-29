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

# we probably only need one of these two lines, may need to delete one eventually?
source ~/.bashrc
module load anaconda3_5.3

conda activate /gscratch/dirac/flair/conda_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gscratch/dirac/flair/conda_env/lib

LK_CACHE_PATH=/gscratch/scrubbed/flair/
SCRIPT_PATH=/gscratch/dirac/flair/pipeline/cvz.py
OUTPUT_PATH=/gscratch/dirac/flair/output/

python $SCRIPT_PATH -o $OUTPUT_PATH -l $LK_CACHE_PATH -t tic272272592 -i 0