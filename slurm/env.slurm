#!/bin/bash
## Job Name
#SBATCH --job-name=flair-env
#SBATCH --account=astro
#SBATCH --partition=compute-bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=3:00:00
#SBATCH --mem=64G
#SBATCH -o /gscratch/dirac/flair/slurm/logs/env_%a_%A.out
#SBATCH -e /gscratch/dirac/flair/slurm/logs/env_%a_%A.err
#SBATCH --chdir=/gscratch/dirac/flair/
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tomwagg@uw.edu          ## <--- CHANGE THIS TO YOUR EMAIL!!!
#SBATCH --export=all

source ~/.bashrc

conda env create -f conda_env.yml
conda activate flair

cd stella_repo
pip install .
