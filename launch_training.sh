#!/bin/bash
#SBATCH --job-name=brainmvp_survival_train
#SBATCH --output=path/finename..out
#SBATCH --error=path/filename.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00

srun python driver.py --config_file path/to/config/file.yaml
