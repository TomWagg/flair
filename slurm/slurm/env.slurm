#!/bin/bash
## Job Name
#SBATCH --job-name=flair-env
#SBATCH --account=astro-ckpt
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3:00:00
#SBATCH --mem=20G
#SBATCH -o /gscratch/astro/wainer/Flare_Project/CVZ/flair/slurm/logs/env_%a_%A.out
#SBATCH -e /gscratch/astro/wainer/Flare_Project/CVZ/flair/slurm/logs/env_%a_%A.err
#SBATCH --chdir=/gscratch/astro/wainer/Flare_Project/CVZ/flair/
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobinw@uw.edu          ## <--- CHANGE THIS TO YOUR EMAIL!!!
#SBATCH --export=all

source ~/.bashrc

conda env create -f conda_env.yml
conda activate flair

cd stella_repo
pip install .
