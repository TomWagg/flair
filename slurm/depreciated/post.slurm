#!/bin/bash
## Job Name
#SBATCH --job-name=post
#SBATCH --account=astro
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=25G
#SBATCH -o /gscratch/dirac/flair/slurm/logs/post_.out
#SBATCH -e /gscratch/dirac/flair/slurm/logs/post_.err
#SBATCH --chdir=/gscratch/dirac/flair/
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobinw@uw.edu          ## <--- CHANGE THIS TO YOUR EMAIL!!!
#SBATCH --export=all

source ~/.bashrc

conda activate /gscratch/dirac/flair/miniconda3/envs/flair
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gscratch/dirac/flair/miniconda3/lib 

CACHE_PATH=/gscratch/scrubbed/tobinw/flair/inject_files/
SCRIPT_PATH=/gscratch/dirac/flair/post_processing/post.py
OUTPUT_PATH=/gscratch/dirac/flair/output/
WRITE_PATH=/gscratch/dirac/flair/Final_Outputs/

python $SCRIPT_PATH -p $OUTPUT_PATH -wp $WRITE_PATH && (mv output/*.h5 $CACHE_PATH) 

