#!/bin/bash

source ~/.bashrc

FLAIR_TIC_ID=$1
SECTOR_ID=$2

conda activate /mnt/home/twainer/miniforge3/envs/flair
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/home/twainer/miniforge3/lib 

CACHE_PATH=/mnt/home/twainer/ceph/scrubb_flair/
SCRIPT_PATH=/mnt/home/twainer/ceph/flair/pipeline/cvz.py
OUTPUT_PATH=/mnt/home/twainer/ceph/flair/output/

echo "Running on ${FLAIR_TIC_ID}, sector ${SECTOR_ID}"

python $SCRIPT_PATH -o $OUTPUT_PATH -c $CACHE_PATH -t $FLAIR_TIC_ID -i $SECTOR_ID