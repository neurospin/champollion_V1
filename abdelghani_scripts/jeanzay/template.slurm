#!/bin/bash
#SBATCH --job-name=job_sigma___SIGMA___factor___FACTOR___batch___BATCH_SIZE___
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=../../../Output/logs/__DATASET___sigma___SIGMA___factor___FACTOR___batch___BATCH_SIZE___%j.out
#SBATCH --error=../../../Output/logs/__DATASET___sigma___SIGMA___factor___FACTOR___batch___BATCH_SIZE___%j.err
#SBATCH -A tgu@h100

module purge

dataset=julien/UKB40/array_load/__DATASET__

set -x

# prepare output directory for logs (will be replaced by sed)
cd "$SCRATCH/Runs/02_champollion_v1/Output/thickness_freesurfer"
mkdir -p "__DATASET___32___BATCH_SIZE__"

# enter the code directory
cd "$SCRATCH/Runs/02_champollion_v1/Program/abdelghani_branch2"
. env_jeanzay2/bin/activate
cd contrastive

python3 train.py \
  +dataset_localization=jeanzay_Run70 \
  +dataset="$dataset" \
  sigma=__SIGMA__ \
  factor=__FACTOR__ \
  batch_size=__BATCH_SIZE__ \
  output_dir="$SCRATCH/Runs/02_champollion_v1/Output/thickness_freesurfer/__DATASET___32___BATCH_SIZE__/sigma___SIGMA___factor___FACTOR___batch___BATCH_SIZE__" \
  -m
