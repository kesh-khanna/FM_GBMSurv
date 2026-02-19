# GBMSurv: Leveraging Foundation Models for Survival Prediction in Glioblastoma with MRIs

A deep learning framework for performing survival prediction in GBM using multimodal MRIs and pretrained feature extactors.

## Overview 
This repo implements end-to-end survival prediction using structural MRI scans (T1,T2-FLAIR,T1GD,T2). The framework currently supports two pretrained backbone architectures:

+ **BrainSegFounder[Paper](https://www.sciencedirect.com/science/article/pii/S1361841524002263)[GitHub](https://github.com/lab-smile/BrainSegFounder)**: The BrainSegFounder is a multi-modal 3D MRI foundation model, built with a SwinVit encoder. This model was pretrained on unlabeled volumes from healthy participants at the UK Biobank and unlabeled volumes from patients with gliomas in the [BraTS challenge dataset](https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/)

+ **BrainMVP[Paper](https://arxiv.org/abs/2410.10604)[GitHub](https://github.com/shaohao011/BrainMVP)**: The BrainMVP is a brain-focused foundation model based on the Uniformer architecture that utilizes a multi-modal contrastive learning framework across 16,022 brain MRI scans from multiple different datasets and disease types.

## Features

+ Multi-modal MRI processing (4-channel input: T1, FLAIR, T1GD, T2)
+ Support for multiple pretrained backbones (BrainMVP, BrainSegFounder)
+ Support for multiple feature map pooling strategies (gap, max, gem)
+ Flexible fine-tuning strategies with backbone layer freezing
+ Survival analysis with Cox proportional hazards loss
+ Multiple ROI sampling strategies (random, tumor-centered, segmentation-weighted)

## Installation

```bash
# Clone the repo
git clone https://github.com/kesh-khanna/FM_GBMSurv.git
cd gbm-survival

# Install dependencies
pip install -r requirements.txt
```

## Pre-trained Weights
+ linked below or on original model GitHubs

Download the pre-trained weights for your chosen backbone:

**BrainMVP**: [BrainMVP Uniformer](https://drive.google.com/file/d/1DTmz5WACESD0wfkZ2r0x-zjTwOgd9ov3/view)

**BrainSegFounder**: [Google Drive](https://drive.google.com/drive/folders/1fl3FeMEhv_cnIwrDa5geHPbKL-tHAuQE)
+ For Glioma specific pretraining, download weights from /BraTS/ssl
+ Choose any one of the available folds

Place the weights in your desired directory and update the `pretrained_weights` path in your configuration file.

## Configuration

The framework uses YAML configuration files. All experiment parameters are specified through these config files. This includes the model architecture, model component freezing strategies, training hyperparameters, and preprocessing approaches. 

We attempted to make the configurations matching, but some parameters are model dependent and documented in the two provided base configs.

### Key Configuration Parameters

**Data Parameters:**
+ `batch_size`: Training batch size 
+ `train_roi_type`: ROI sampling strategy during training (`random`, `seg_weighted`)
+ `val_roi_type`: ROI sampling for validation (`center_crop`, `seg_weighted`)
+ `normalization_method`: Normalization approach (`z_score` or `percentile` for 5-95 percentile scaling)
+ `train_cache_rate`/`val_cache_rate`: MONAI cache rate (0.0-1.0) for caching of non-random transforms
+ `train_patch_shape` and `val_patch_shape`: Determines the size of ROIs

**Model Parameters:**
+ `type`: Model architecture (`brainmvp` or `brainseg`)
+ `pretrained_weights`: Path to pretrained backbone weights
+ `use_pretrained_weights`: Whether or not to load the provided pretraining weights
+ `checkpoint_path`: Path to checkpoint for prediction-only mode
+ Fine-tuning controls: `train_patch_embed`, `train_stage1`, etc. control which layers are trainable

**Training Parameters:**
+ `backbone_lr`/`head_lr`: Separate learning rates for backbone and prediction head
+ `evaluation_strategy`: `last_epoch` or `best_val_cindex` (enables early stopping)
+ `new_time`: Time horizon in days for AUC calculation (e.g., 365.0 for 1-year survival)

## Usage

### Training

Train a model using a configuration file:

```bash
python driver.py --config_file configs/brainmvp_config.yaml
```

**Optional Arguments:**
+ `--disable_progress_bar`: Disable progress bars during training and validation (useful for cluster jobs)

### Prediction Only Mode

Run inference on test data using a given checkpoint:

```bash
python driver.py --config_file configs/brainmvp_config.yaml --predict_only
```

When using `--predict_only`, make sure your config file includes the desired checkpoint path:

```yaml
model:
  checkpoint_path: "/path/to/your/checkpoint.ckpt"
```

### Output Structure

Training creates the following directory structure:

```
output/
└── your_experiment_save_name/
    ├── config.yaml                    # Saved configuration
    ├── results_summary.json           # Final evaluation metrics
    ├── tensorboard/                   # TensorBoard logs
    ├── checkpoints/                   # Model checkpoints
    │   ├── model_epoch_X_val_cindex_Y.ckpt  # Best models (if using best_val_cindex)
    │   ├── last_epoch_Z.ckpt         # Last epoch checkpoint
    │   └── epoch_N.ckpt              # Regular checkpoints
    └── predictions/                   # Prediction CSVs (if specified)
        ├── {save_name}_train_predictions.csv
        ├── {save_name}_val_predictions.csv
        └── {save_name}_test_predictions.csv
```
### Evaluation Results

After training completes, the framework:

1. **Automatically evaluates** on all available splits (train/validation/test)
2. **Saves predictions** to CSV files (if `prediction_dir` is specified)
3. **Generates a results summary** (`results_summary.json`) containing:
   + C-Index with 95% confidence intervals
   + AUC with 95% confidence intervals
   + Checkpoint used for evaluation

**Example results_summary.json:**
```json
{
  "eval_checkpoint": "output/experiment/checkpoints/model_epoch_20_val_cindex_0.725.ckpt",
  "test_results": {
    "Test_auc": 0.7234,
    "Test_c_index": 0.7156,
    "Test_auc_ci": [0.6823, 0.7645],
    "Test_c_index_ci": [0.6734, 0.7578]
  },
  "train_results": {...},
  "val_results": {...}
}
```

**Prediction CSV format:**
```csv
patient_id,log_hz,time,event
PATIENT_001,0.234,1039.0,True
PATIENT_002,0.567,709.0,True
PATIENT_003,1.123,450.0,False
```

Where:
+ `log_hz`: Relative hazard prediction
+ `time`: Survival time in days
+ `event`: True if event occurred, False if censored

**Additional Logged Metrics (Tensorboard):**
+ Loss (train/validation)
+ C-Index (train/validation)
+ AUC (train/validation)
+ Learning rates
+ Gradient norms (backbone, head, total, and ratios)

## Fine-tuning Strategies

The framework supports flexible fine-tuning approaches:

### Full Fine-tuning
All layers are trainable (default, or when layer-specific controls are set to `True`)

### Partial Fine-tuning
i.e. Freeze early layers, train later ones:
```yaml
# brainsegfounder specific layer names
model:
  train_patch_embed: False
  train_stage1: False
  train_stage2: True
  train_stage3: True
  train_stage4: True
```

## ROI Sampling Strategies

### Training
- **`random`**: Random crop from the foreground cropped volume
- **`seg_weighted`**: Weighted sampling based on tumor segmentation mask

### Validation/Testing
+ **`center_crop`**: Center crop of specified size from foreground cropped volume
+ **`tumor_centered`**: Crop centered on tumor centroid

## A Few Tips and Best Practices

1. **Batch Size**: Larger batches provide better within-batch risk estimations. This is especially the case when the event-rate is low.

2. **Learning Rates**: 
   + If fully finetuning, finetune backbone lightly.
   + i.e. backbone_lr=1e-5, head_lr=1e-4

3. **Normalization**: 
   + BrainMVP: was trained with `percentile`
   + BrainSegFounder: was trained with `z_score`


## Citation
+ to be added upon publication

## License
+ to be added

## Additional Acknowledgements and Thanks To:
+ BrainSegFounder Team
+ BrainMVP Team
+ MONAI Framework Team
+ TorchSurv Library Team