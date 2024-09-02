#!/bin/bash
## Job Name
#SBATCH --job-name=cvz-flair
#SBATCH --account=astro
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:00
#SBATCH --mem=50G
#SBATCH -o /gscratch/dirac/flair/slurm/logs/cvz_%A_%a.out
#SBATCH -e /gscratch/dirac/flair/slurm/logs/cvz_%A_%a.err
#SBATCH --chdir=/gscratch/dirac/flair/pipeline
##SBATCH --mail-type=ALL
##SBATCH --mail-user=tobinw@uw.edu          ## <--- CHANGE THIS TO YOUR EMAIL!!!
#SBATCH --export=all

source ~/.bashrc

conda activate /gscratch/dirac/flair/miniconda3/envs/flair
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gscratch/dirac/flair/miniconda3/lib 

CACHE_PATH=/gscratch/scrubbed/tobinw/flair/
SCRIPT_PATH=/gscratch/dirac/flair/pipeline/cvz.py
OUTPUT_PATH=/gscratch/dirac/flair/output/

echo "Running on ${FLAIR_TIC_ID}, sector ${SLURM_ARRAY_TASK_ID}"

python $SCRIPT_PATH -o $OUTPUT_PATH -c $CACHE_PATH -t $FLAIR_TIC_ID -i $SLURM_ARRAY_TASK_ID -p 1 && python check_failed.py
