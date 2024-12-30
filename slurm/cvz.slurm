#!/bin/bash
## Job Name
#SBATCH --job-name=Flair
##SBATCH --account=cca
#SBATCH --partition=cca
#SBATCH --nodes=8
##SBATCH --ntasks-per-node=32
#SBATCH --time=55:00:00
##SBATCH --mem=50G
#SBATCH -o /mnt/home/twainer/ceph/flair/slurm/logs/slurm_%j.out
#SBATCH -e /mnt/home/twainer/ceph/flair/slurm/logs/slurm_%j.err
#SBATCH --chdir=/mnt/home/twainer/ceph/flair/slurm/
#SBATCH --mail-type=ALL
##SBATCH --mail-user=tobinw@uw.edu          ## <--- CHANGE THIS TO YOUR EMAIL!!!
#SBATCH --export=all

source ~/.bashrc

conda activate /mnt/home/twainer/miniforge3/envs/flair

module load disBatch

disBatch full_disbatch 
