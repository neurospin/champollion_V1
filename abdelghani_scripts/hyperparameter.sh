#!/bin/bash

SIGMA_LIST="0.001 0.005 0.01 0.02 0.03 0.05 0.075 0.1"
BATCH_LIST="128"
DATASET_LIST="SC-sylv_right_UKB40"
OUTPUT_DIR="generated_jobs"

mkdir -p $OUTPUT_DIR

for DATASET in $DATASET_LIST; do
  for SIGMA in $SIGMA_LIST; do
    for BATCH in $BATCH_LIST; do
      JOB_NAME="${DATASET}_sigma_${SIGMA}_batch_${BATCH}"
      JOB_FILE="${OUTPUT_DIR}/${JOB_NAME}.slurm"

      sed -e "s/XX_SIGMA/$SIGMA/g" \
          -e "s/XX_BATCH_SIZE/$BATCH/g" \
          -e "s|XX_DATASET|$DATASET|g" \
          template.slurm > "$JOB_FILE"

      echo "Submitting job $JOB_FILE"
      sbatch "$JOB_FILE"
    done
  done
done

