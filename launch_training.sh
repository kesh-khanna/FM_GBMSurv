#!/bin/bash
#SBATCH --job-name=brainmvp_survival_train
#SBATCH --output=/data/nextgen/keshk/foundation_models/FM_GBMSurv/out/nci_test_brainmvp/finetune_aug/%j.out
#SBATCH --error=/data/nextgen/keshk/foundation_models/FM_GBMSurv/out/nci_test_brainmvp/finetune_aug/%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00

srun python driver.py --config_file /data/nextgen/keshk/foundation_models/FM_GBMSurv/configs/BrainMVP/test_config.yaml  --disable_progress_bar
