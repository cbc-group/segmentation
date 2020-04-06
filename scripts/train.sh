#!/bin/bash
#PBS -N train
#PBS -j oe
#PBS -m abe

# prepare path
source "${conda_base}/etc/profile.d/conda.sh"

# launch environment
conda activate segmentation

train3dunet --config ${config_path}
