# Readme

## Overview
This directory contains shell scripts (`.sh`) and SLURM batch scripts for running training and ablation study experiments. The scripts are organized to handle multiple seeds for generalization purposes.

## Contents
- **Training Scripts**: Shell and SLURM scripts for running the main training experiments.
- **Ablation Study Scripts**: Shell and SLURM scripts specifically designed for ablation studies, with support for running experiments per seed.

## Usage
1. **Training**:
    - Use the provided `.sh` and SLURM scripts to launch training jobs.
    - Ensure to specify the desired seed for generalization experiments.

2. **Ablation Study**:
    - Run the ablation study scripts to analyze the impact of specific components.
    - Each script supports running experiments per seed.

## Notes
- Modify the SLURM parameters (e.g., time, memory, partition) as needed for your computing environment.
- Modify the Sh variables (e.g., dataset, sigma_factor, batch_size) to specify the ablation study parameters.
- Ensure all dependencies are installed and paths are correctly set before running the scripts.
