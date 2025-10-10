#!/bin/bash

SIGMA="null"                # always null
FACTOR_LIST="0.2 0.4"       # sweep values
BATCH_LIST="16"
DATASET_LIST="SOr_left_UKB40"
OUTPUT_DIR="generated_jobs"

mkdir -p "$OUTPUT_DIR"

for DATASET in $DATASET_LIST; do
  for FACTOR in $FACTOR_LIST; do
    for BATCH in $BATCH_LIST; do
      JOB_NAME="${DATASET}_sigma_${SIGMA}_factor_${FACTOR}_batch_${BATCH}"
      JOB_FILE="${OUTPUT_DIR}/${JOB_NAME}.slurm"

      sed -e "s/__SIGMA__/${SIGMA}/g" \
          -e "s/__FACTOR__/${FACTOR}/g" \
          -e "s/__BATCH_SIZE__/${BATCH}/g" \
          -e "s|__DATASET__|${DATASET}|g" \
          template.slurm > "$JOB_FILE"

      echo "Submitting job $JOB_FILE"
      sbatch "$JOB_FILE"
    done
  done
done
