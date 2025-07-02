#!/bin/bash
#SBATCH --job-name=job_sigma_XX_SIGMA_batch_XX_BATCH_SIZE
#SBATCH -C a100    
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=../../../Output/logs/XX_DATASET_sigma_XX_SIGMA_batch_XX_BATCH_SIZE_%j.out
#SBATCH --error=../../../Output/logs/XX_DATASET_sigma_XX_SIGMA_batch_XX_BATCH_SIZE_%j.err
#SBATCH -A tgu@a100

module purge

dataset=julien/UKB40/array_load/XX_DATASET

set -x

cd $SCRATCH
cd Runs/02_champollion_v1/Output/PGS
mkdir -p XX_DATASET
cd $SCRATCH
cd Runs/02_champollion_v1/Program/2023_jlaval_STSbabies/
. env_jeanzay2/bin/activate
cd contrastive/

python3 train.py \
  +dataset_localization=jeanzay_Run70 \
  +dataset=$dataset \
  sigma=XX_SIGMA \
  batch_size=XX_BATCH_SIZE \
  output_dir=$SCRATCH/Runs/02_champollion_v1/Output/PGS/XX_DATASET/sigma_XX_SIGMA_batch_XX_BATCH_SIZE \
  -m

