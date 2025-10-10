#!/bin/bash


# SIGMA_LIST="null"
# SIGMA_FACTOR_LIST="0.05 0.1 0.2 0.4 0.6 0.8 1.0 1.2 1.4"

SIGMA="null"
SIGMA_FACTOR="0.2"

BATCH_LIST="16"
DATASET_LIST="FCLp-subsc-FCLa-INSULA_left OCCIPITAL_left"

TRAIN_SEEDS=("42" "75" "48" "128" "212")

OUTPUT_DIR="generated_jobs"
mkdir -p "$OUTPUT_DIR"

for DATASET in $DATASET_LIST; do
  for BATCH in $BATCH_LIST; do
    for SEED in "${TRAIN_SEEDS[@]}"; do

      JOB_NAME="${DATASET}_sigma_${SIGMA}_factor_${SIGMA_FACTOR}_batch_${BATCH}_seed_${SEED}"
      JOB_FILE="${OUTPUT_DIR}/${JOB_NAME}.slurm"

      sed -e "s/XX_SIGMA_FACTOR/${SIGMA_FACTOR}/g" \
          -e "s/XX_SIGMA/${SIGMA}/g" \
          -e "s/XX_BATCH_SIZE/${BATCH}/g" \
          -e "s/XX_DATASET/${DATASET}/g" \
          -e "s/XX_TRAIN_SEED/${SEED}/g" \
          seed_template.slurm > "$JOB_FILE"

      echo "Submitting job $JOB_FILE"
      sbatch "$JOB_FILE"

    done
  done
done
