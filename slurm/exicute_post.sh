#!/bin/bash

source ~/.bashrc

conda activate /mnt/home/twainer/miniforge3/envs/flair
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/home/twainer/miniforge3/lib 

CACHE_PATH=/mnt/home/twainer/ceph/scrubb_flair/inject_files/
CACHE_PATH2=/mnt/home/twainer/ceph/scrubb_flair/log_files/
CACHE_PATH3=/mnt/home/twainer/ceph/scrubb_flair/img_files/
SCRIPT_PATH=/mnt/home/twainer/ceph/flair/pipeline/cvz.py
OUTPUT_PATH=/mnt/home/twainer/ceph/output/

SCRIPT_PATH=/mnt/home/twainer/ceph/flair/post_processing/post.py
OUTPUT_PATH=/mnt/home/twainer/ceph/flair/output/
WRITE_PATH=/mnt/home/twainer/ceph/flair/Final_Outputs/
WRITE_PATH2=/mnt/home/twainer/ceph/flair/Final_Outputs/initial_plots/

python $SCRIPT_PATH -p $OUTPUT_PATH -wp $WRITE_PATH && (mv ../output/*.h5 $CACHE_PATH) && (mv logs/tic* $CACHE_PATH2) && (mv ../output/plots/*.png $CACHE_PATH3)