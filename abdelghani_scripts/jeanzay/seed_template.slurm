#!/bin/bash
#SBATCH --job-name=job_sigma_XX_SIGMA_factor_XX_SIGMA_FACTOR_batch_XX_BATCH_SIZE_seed_XX_TRAIN_SEED
#SBATCH -C v100    
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=../../../Output/logs/XX_DATASET_sigma_XX_SIGMA_factor_XX_SIGMA_FACTOR_batch_XX_BATCH_SIZE_seed_XX_TRAIN_SEED_%j.out
#SBATCH --error=../../../Output/logs/XX_DATASET_sigma_XX_SIGMA_factor_XX_SIGMA_FACTOR_batch_XX_BATCH_SIZE_seed_XX_TRAIN_SEED_%j.err
#SBATCH -A tgu@v100

module purge

dataset=julien/UKB40/array_load/XX_DATASET

set -x


cd $SCRATCH/Runs/02_champollion_v1/Output/MAHLANOBIS_reg
mkdir -p XX_DATASET_16_XX_BATCH_SIZE


cd $SCRATCH/Runs/02_champollion_v1/Program/abdelghani_branch2
. env_jeanzay2/bin/activate


python3 -m contrastive.diff_seed_train \
  +dataset_localization=jeanzay_Run70 \
  +dataset=$dataset \
  sigma=XX_SIGMA \
  factor=XX_SIGMA_FACTOR \
  batch_size=XX_BATCH_SIZE \
  train_seed=XX_TRAIN_SEED \
  output_dir=$SCRATCH/Runs/02_champollion_v1/Output/MAHLANOBIS_reg/XX_DATASET_16_XX_BATCH_SIZE/batch_XX_BATCH_SIZE_seed_XX_TRAIN_SEED \
  -m
