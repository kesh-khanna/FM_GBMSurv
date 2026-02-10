"""
Author: Rakesh Khanna
"""
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from monai.data import set_track_meta
import time
import os
import random
import monai.transforms as transforms
from monai.transforms import ToTensord
from monai.data import Dataset, CacheDataset, DataLoader

from torchsurv.loss.cox import neg_partial_log_likelihood
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.cindex import ConcordanceIndex
import yaml
from tqdm import tqdm
import logging
from torch.amp import GradScaler
import json
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
print(sys.executable)
import monai
print(monai.__version__)
import torch
print(torch.__version__)
import torch.nn as nn

from create_model import create_model
from transforms.transforms import custom_transform
from utils.torch_utils import set_seed, clear_memory, set_bn_eval

def create_optimizer_scheduler(model, config):
    """
    Create optimizer + scheduler with two learning rates:
      - backbone_lr: model.embedder.encoder (i.e. UniFormer)
      - head_lr: everything else (pooling + survival head)
    Only includes params with requires_grad=True.
    """

    wd = float(config["training"]["reg_weight"])
    backbone_lr = float(config["training"].get("backbone_lr", config["training"].get("optim_lr", 3e-4)))
    head_lr = float(config["training"].get("head_lr", config["training"].get("optim_lr", 1e-3)))
    optim_name = config["training"]["optim_name"].lower()

    # Split params
    encoder_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("embedder.encoder"):
            encoder_params.append(p)
        else:
            head_params.append(p)

    print(f"Using two LRs: backbone_lr={backbone_lr} head_lr={head_lr} wd={wd} optim={optim_name}")
    print(f"Trainable params: encoder={sum(p.numel() for p in encoder_params):,} "
          f"head={sum(p.numel() for p in head_params):,}")

    param_groups = [
        {"params": encoder_params, "lr": backbone_lr, "weight_decay": wd},
        {"params": head_params, "lr": head_lr, "weight_decay": wd},
    ]

    # Optimizer
    if optim_name == "adam":
        optimizer = torch.optim.Adam(param_groups)
    elif optim_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups)
    else:
        raise ValueError(f"Unsupported Optimization Procedure: {optim_name}")

    # Scheduler
    lrschedule = config["training"].get("lrscheduler", None)
    if lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=config["training"]["warmup_epochs"],
            max_epochs=config["training"]["max_epochs"],
        )
    elif lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["training"]["max_epochs"]
        )
        if config.get("training", {}).get("checkpoint", None) is not None:
            scheduler.step(epoch=0)
    else:
        scheduler = None

    return optimizer, scheduler


class ModelTrainer:
    def __init__(self, model, device, config, output_dir):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.output_dir = output_dir
        
        self.max_epochs = config["training"]["max_epochs"]
        self.heading_lr = config["training"]["head_lr"]
        self.backbone_lr = config["training"]["backbone_lr"]
        self.weight_decay = config["training"]["reg_weight"]
        
        # validation settings
        self.validation_mode = config["training"].get("validation_mode", "early_stopping")
        self.use_last_model = config["training"].get("use_last_model", False)
        self.patience = config["training"].get("patience", 20)
        
        # Validation modes: "early_stopping", "monitor_only", "none"
        assert self.validation_mode in ["early_stopping", "monitor_only", "none"], \
            f"validation_mode must be one of: early_stopping, monitor_only, none. Got: {self.validation_mode}"

        # testing out mixed precision. generally safe and may let us use a larger batch size
        self.scaler = GradScaler(enabled=self.config["training"]["mixed_precision"])

        # how often to save a standard checkpoint
        self.checkpoint_frequency = config["training"]["checkpoint_frequency"]
        self.save_top_k = config["training"].get("save_top_k", None)  # none means save all
        self.saved_checkpoints = []

        self.optimizer, self.scheduler = create_optimizer_scheduler(self.model, config)

        self.loss_module = neg_partial_log_likelihood
        
        # time horizon for AUC
        self.new_time = torch.tensor(config["training"]["new_time"])

        print(f"Using new_time: {self.new_time.item()} days for AUC calculation")
        
        # tracking variables
        self.best_score = -float('inf')

        self.patience_counter = 0
        self.global_step = 0
        self.epoch = 0

        # setup tensorboard
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))
        
        # create checkpoint directory
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.init_metrics()
    
    def init_metrics(self):
        """
        Initalize metrics
        Keep separate to ensure confidence intervals are done properly
        """
        # Training metrics
        self.training_auc = Auc()
        self.training_c = ConcordanceIndex()
        
        # Validation metrics
        self.val_auc = Auc()
        self.val_c = ConcordanceIndex()
        
        # Test metrics
        self.testing_auc = Auc()
        self.testing_c = ConcordanceIndex()

    def save_checkpoint(self, is_best=False, is_last=False):
        """
        save model checkpoint and relevant states if they are available
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'best_score': self.best_score,
            'global_step': self.global_step,
            'config': self.config,
            'scaler': self.scaler.state_dict() if hasattr(self, 'scaler') else None
        }
        
        if is_best:
            # denote in filename that this is the best model with epoch and validation C-Index
            checkpoint_path = os.path.join(self.checkpoint_dir, f'best_model_epoch_{self.epoch}_val_cindex_{self.best_score:.3f}.ckpt')

            # add to our tracked checkpoints
            self.saved_checkpoints.append(checkpoint_path)

            if self.save_top_k is not None and len(self.saved_checkpoints) > self.save_top_k:
                # remove the oldest checkpoint if we exceed the limit
                oldest_checkpoint = self.saved_checkpoints.pop(0)
                if os.path.exists(oldest_checkpoint):
                    os.remove(oldest_checkpoint)
                    logger.info(f"Removed old checkpoint: {oldest_checkpoint}")

        elif self.epoch == self.max_epochs - 1:
            # this is the last epoch checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, f'last_epoch_{self.epoch}.ckpt')

        elif is_last:
            # this is the last epoch checkpoint due to early stopping
            print("Saving last epoch checkpoint due to early stopping")
            checkpoint_path = os.path.join(self.checkpoint_dir, f'last_epoch_{self.epoch}.ckpt')

        else:
            # this is a regular checkpoint 
            checkpoint_path = os.path.join(self.checkpoint_dir, f'epoch_{self.epoch}.ckpt')

        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Checkpoint saved at epoch {self.epoch}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint {checkpoint_path} not found")
            return False
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])

        self.best_score = checkpoint['best_score']
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Model loaded from: epoch {self.epoch}, global step {self.global_step}")

        if "round" in checkpoint:
            logger.info(f"Round loaded from checkpoint: {checkpoint['round']}")

        return True
    
    def train_epoch(self, train_loader, disable_pbar=False):
        """
        Train for one epoch.
        """
        self.model.train()

        # if we want to freeze the batchnorm then we have to do it at every epoch
        if self.config["model"].get("freeze_batchnorm", False):
            self.model.apply(set_bn_eval)

        total_loss = 0.0
        num_loss_computations = 0

        all_log_hz, all_time, all_events = [], [], []

        for batch_idx, batch in enumerate(tqdm(train_loader, desc='Training', disable=disable_pbar)):
            try:
                images = batch['image'].to(self.device, non_blocking=True)
                time = batch['label'].to(self.device, non_blocking=True)
                events = batch['event'].bool().to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=self.config["training"]["mixed_precision"]):
                    log_hz = self.model(images)  
                    # print(log_hz.shape, log_hz.dtype)

                # Save detached copies for epoch metrics
                all_log_hz.append(log_hz.detach().cpu())
                all_time.append(time.detach().cpu())
                all_events.append(events.detach().cpu())

                # Cox loss expects float64 for stability
                with torch.amp.autocast("cuda", enabled=False):
                    log_hz64 = log_hz.double()
                    loss = self.loss_module(
                        log_hz64,
                        event=events,
                        time=time,
                        reduction="mean"
                    )

                # skip if no events in this mini-batch. Consider stratified batch sampling if event rate is low
                if events.sum() == 0:
                    if not disable_pbar:
                        tqdm.write(f"Batch {batch_idx+1}: no events, skipping update")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                # Backward + step
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                num_loss_computations += 1
                self.global_step += 1

                if not disable_pbar:
                    tqdm.write(f"Epoch {self.epoch}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

                if batch_idx == 0:
                    print("images:", images.shape, images.dtype)
                    print("log_hz:", log_hz.shape, log_hz.dtype)
                    print("allocated GB:", torch.cuda.memory_allocated()/1e9)
                    print("reserved  GB:", torch.cuda.memory_reserved()/1e9)

                del images, time, events, log_hz, log_hz64, loss

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"CUDA OOM at batch {batch_idx}")
                    clear_memory()
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                raise

        avg_loss = total_loss / num_loss_computations if num_loss_computations > 0 else float('inf')

        # Epoch-level metrics
        if all_log_hz:
            all_log_hz = torch.cat(all_log_hz, dim=0)
            all_time = torch.cat(all_time, dim=0)
            all_events = torch.cat(all_events, dim=0).bool()

            new_time = self.new_time
            train_auc = self.training_auc(all_log_hz, all_events, all_time, new_time=new_time)
            train_c = self.training_c(all_log_hz, all_events, all_time)

            del all_log_hz, all_time, all_events
        else:
            train_auc = 0.0
            train_c = 0.0

        # Log metrics
        self.writer.add_scalar('Loss/Train', avg_loss, self.epoch)
        self.writer.add_scalar('AUC/Train', train_auc.item() if hasattr(train_auc, "item") else float(train_auc), self.epoch)
        self.writer.add_scalar('C-Index/Train', train_c.item() if hasattr(train_c, "item") else float(train_c), self.epoch)
        for i, g in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'LR/group_{i}', g['lr'], self.epoch)

        logger.info(f"Train - Avg Loss: {avg_loss:.4f}, AUC: {float(train_auc):.4f}, C-Index: {float(train_c):.4f}")
        return avg_loss

    
    def validate_full_dataset(self, val_loader, disable_pbar=False):
        """
        Validate on entire dataset at once for survival prediction.
        This accumulates all predictions and targets before computing loss.
        """
        self.model.eval()
        
        all_log_hz = []
        all_time = []
        all_events = []
        
        logger.info("collecting predictions for validation...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validation', disable=disable_pbar)):
                try:
                    images = batch['image'].to(self.device, non_blocking=True)
                    time = batch['label'].to(self.device, non_blocking=True)
                    events = batch['event'].bool().to(self.device, non_blocking=True)

                    with torch.amp.autocast("cuda", enabled=self.config["training"]["mixed_precision"]):
                        log_hz = self.model(images)
                    
                    all_log_hz.append(log_hz.cpu())
                    all_time.append(time.cpu())
                    all_events.append(events.cpu())
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error(f"CUDA OOM during validation at batch {batch_idx}")
                        clear_memory()
                        continue
                    else:
                        raise e
        
        if all_log_hz:
            # cat all predictions and targets
            all_log_hz = torch.cat(all_log_hz, dim=0)
            all_time = torch.cat(all_time, dim=0)
            all_events = torch.cat(all_events, dim=0).bool()
            new_time = self.new_time
            
            # get the loss on the entire validation set
            # again dont use amp since the loss doesnt accept half precision. Use float 64 instead to be safe
            with torch.amp.autocast("cuda", enabled=False):
                # TODO: right now the loss for the validation set is not directly comparable to the training loss
                # in the training loss is computed on the effective batch size
                # here we compute the loss on the full validation set
                # this is not a problem if the effective batch size is equal to the full batch size
                # but likely it will not be
                # might need to scale the validation loss by the effective batch size for plotting and comparison but will test
                all_log_hz = all_log_hz.double()
                val_loss = self.loss_module(all_log_hz, event=all_events, time=all_time, reduction="mean").item()
            
            # calculate the metrics
            val_auc = self.val_auc(all_log_hz, all_events, all_time, new_time=new_time)
            val_c = self.val_c(all_log_hz, all_events, all_time)    
            
            # empty memory after validation
            del all_log_hz, all_time, all_events

        else:
            val_loss = float('inf')
            val_auc = 0.0
            val_c = 0.0
        
        # Log metrics
        self.writer.add_scalar('Loss/Validation', val_loss, self.epoch)
        self.writer.add_scalar('AUC/Validation', val_auc.item(), self.epoch)
        self.writer.add_scalar('C-Index/Validation', val_c.item(), self.epoch)
        
        logger.info(f"Val - Loss: {val_loss:.4f}, AUC: {val_auc.item():.4f}, C-Index: {val_c.item():.4f}")
        
        return val_loss, val_auc.item(), val_c.item()
    
    def train(self, train_loader, val_loader=None, disable_pbar=False):
        """
        Full training loop with validation and early stopping
        Train for max_epochs or until early stopping is triggered
        """
        logger.info(f"starting training for {self.max_epochs} epochs")
        logger.info(f"Validation mode: {self.validation_mode}")
        logger.info(f"Use last model: {self.use_last_model}")
        logger.info(f"using device: {self.device}")
        logger.info(f"Training batch size: {self.config['data']['batch_size']}")
        
        if val_loader is not None:
            logger.info(f"Validation batch size: {self.config['data']['val_batch_size']}")
        
        logger.info(f"Weight Decay: {self.weight_decay}")
        logger.info(f"Encoder LR: {self.backbone_lr}")
        logger.info(f"Heading LR: {self.heading_lr}")


        start_time = time.time()
        
        for epoch in range(self.max_epochs):
            self.epoch = epoch
            
            # Training
            self.train_epoch(train_loader, disable_pbar=disable_pbar)
            
            # step the scheduler
            # I think this should be done every epoch
            if self.scheduler is not None:
                self.scheduler.step()

            # Validation
            if val_loader is not None and self.validation_mode != "none":
                val_loss, val_auc, val_c = self.validate_full_dataset(val_loader, disable_pbar=disable_pbar)
                
                # Track best model (for logging/monitoring)
                monitor = val_c
                is_best = monitor > self.best_score
                
                if is_best:
                    self.best_score = max(monitor, self.best_score)
                    logger.info(f"New best validation C-Index: {val_c:.4f}")
                    
                    # Only save as "best" if using early stopping
                    if self.validation_mode == "early_stopping":
                        self.patience_counter = 0
                        self.save_checkpoint(is_best=True)
                
                # Early stopping (only if mode is "early_stopping")
                if self.validation_mode == "early_stopping":
                    if not is_best:
                        self.patience_counter += 1
                    
                    if self.patience_counter >= self.patience:
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        self.save_checkpoint(is_best=False, is_last=True)
                        break
        
            
            if (epoch + 1) % self.checkpoint_frequency == 0 or (epoch + 1) == self.max_epochs:
                self.save_checkpoint(is_best=False)
            
            # clear the memory after each epoch?
            # Should explore the pros and cons and maybe track memory usage overtime
            # clear_memory()
        
        training_time = (time.time() - start_time) / 60
        logger.info(f"Training completed in {training_time:.2f} minutes")
        
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / 1e9
            logger.info(f"Peak GPU memory usage: {max_memory:.2f} GB")
        
        # close the tensorboard writer
        self.writer.close()
    
    def predict(self, data_loader, checkpoint_path=None, disable_pbar=False):
        """
        make predictions on a dataset
        saves the patient_id so we can match predictions with other clinical data after 
        late fusion
        """
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc='Predicting', disable=disable_pbar)):
                try:
                    images = batch['image'].to(self.device, non_blocking=True)
                    time = batch['label'].to(self.device, non_blocking=True)
                    events = batch['event'].bool().to(self.device, non_blocking=True)
                    patient_id = batch.get('patient_id', None)
                    
                    with torch.amp.autocast("cuda", enabled=self.config["training"]["mixed_precision"]):
                        log_hz = self.model(images)
                    
                    # Convert to predictions format
                    batch_preds = {
                        'log_hz': log_hz.cpu().numpy(),
                        'time': time.cpu().numpy(),
                        'events': events.cpu().numpy(),
                        'patient_id': patient_id,
                    }
                    predictions.append(batch_preds)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error(f"CUDA OOM during prediction at batch {batch_idx}")
                        clear_memory()
                        continue
                    else:
                        raise e
    
        pred_df = pd.DataFrame(predictions)

        return pred_df

    def test(self, test_loader, checkpoint_path=None, disable_pbar=False):
        """
        test the model and compute metrics
        optinally lets us load a checkpoint
        """
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
        
        all_log_hz = []
        all_time = []
        all_events = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc='Testing', disable=disable_pbar)):
                try:
                    images = batch['image'].to(self.device, non_blocking=True)
                    time = batch['label'].to(self.device, non_blocking=True)
                    events = batch['event'].bool().to(self.device, non_blocking=True)

                    with torch.amp.autocast("cuda", enabled=self.config["training"]["mixed_precision"]):
                        log_hz = self.model(images)
                    
                    all_log_hz.append(log_hz.cpu())
                    all_time.append(time.cpu())
                    all_events.append(events.cpu())

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error(f"CUDA OOM during testing at batch {batch_idx}")
                        clear_memory()
                        continue
                    else:
                        raise e
        
        # Concatenate all outputs and compute metrics on entire test set
        # Cast to float to ensure compatibility with AUC and C-Index calculations
        if all_log_hz:
            all_log_hz = torch.cat(all_log_hz, dim=0).float()
            all_time = torch.cat(all_time, dim=0).float()
            all_events = torch.cat(all_events, dim=0).bool()
            
            # get metrics on the entire test set
            test_auc = self.testing_auc(all_log_hz, all_events, all_time, new_time=self.new_time.float())
            test_auc_ci = self.testing_auc.confidence_interval(method="bootstrap")
            test_c = self.testing_c(all_log_hz, all_events, all_time)
            test_c_ci = self.testing_c.confidence_interval(method="bootstrap")
            
            logger.info(f"Test - AUC: {test_auc.item():.4f}, C-Index: {test_c:.4f}")
            logger.info(f"Test - AUC Confidence Interval: {test_auc_ci.tolist()}, C-Index Confidence Interval: {test_c_ci.tolist()}")
            
            # Clear memory
            del all_log_hz, all_time, all_events
        
            return {'test_auc': test_auc.item(), 'test_c_index': test_c.item(), 
                    'test_auc_ci': test_auc_ci.tolist(), 'test_c_index_ci': test_c_ci.tolist()}
        else:
            logger.warning("No test data processed")
            return {'test_auc': 0.0, 'test_c_index': 0.0, "test_auc_ci": None, "test_c_index_ci": None}


def create_parser():
    parser = argparse.ArgumentParser(description="FL_BrainSurViT Driver")
    parser.add_argument("--config_file", required=True, type=str, help="Path to YAML config for the model and training")
    parser.add_argument("--predict_only", action="store_true", help="Only make predictions")
    parser.add_argument("--disable_progress_bar", action="store_true", help="Disable progress bar for training and validation")
    return parser

# Should move these two to utils
def load_config(config_path: str) -> Dict[str, Any]:
    """
    load out config from the yaml, should be structured
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def main():
    parser = create_parser()
    args = parser.parse_args()

    config = load_config(args.config_file)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_track_meta(True)
    
    logger.info(f"using device: {device}")
    logger.info(f"config loaded from: {args.config_file}")

    # load in the json file
    with open(config["data"]["json_file"], "r") as f:
        json_data = json.load(f)

    train_data = json_data.get("train", None)
    val_data = json_data.get("validation", None)
    test_data = json_data.get("test", None)

    # make each of the splits into a list of dicts and move the ID key to inside the dict
    if train_data:
        for patient_id in train_data:
            train_data[patient_id] = {'image': train_data[patient_id]['image'], 'patient_id': patient_id, 'event': int(train_data[patient_id]['event']), 'label': float(train_data[patient_id]['label'])}
        train_data = list(train_data.values())
        print(f"Number of training patients: {len(train_data)}")
        print(f"Example data point: {train_data[0]}")
    if val_data:
        for patient_id in val_data:
            val_data[patient_id] = {'image': val_data[patient_id]['image'], 'patient_id': patient_id, 'event': int(val_data[patient_id]['event']), 'label': float(val_data[patient_id]['label'])}
        val_data = list(val_data.values())
        print(f"Number of validation patients: {len(val_data)}")
    if test_data:
        for patient_id in test_data:
            test_data[patient_id] = {'image': test_data[patient_id]['image'], 'patient_id': patient_id, 'event': int(test_data[patient_id]['event']), 'label': float(test_data[patient_id]['label'])}
        test_data = list(test_data.values())
        print(f"Number of test patients: {len(test_data)}")
    
    # check the amount of censoring in each split
    def check_censoring(data, split_name):
        if data:
            num_events = sum([d['event'] for d in data])
            num_censored = len(data) - num_events
            censoring_rate = num_censored / len(data)
            print(f"{split_name} - Total: {len(data)}, Events: {num_events}, Censored: {num_censored}, Censoring Rate: {censoring_rate:.2f}")
        else:
            print(f"{split_name} data is empty or not provided.")

    check_censoring(train_data, "Training")
    check_censoring(val_data, "Validation")
    check_censoring(test_data, "Testing")

    train_transforms, val_transforms = custom_transform(config)

    # cache 50% of the data for training and validation to speed up training
    # can increase this if we have enough memory
    # for training we add augmentations after caching so that we dont cache augmented data
    # for validation we just use the validation transforms
    # for testing we also use the validation transforms
    # if we have no training data (predict only mode) then we skip the training ds
    train_cache_rate = config["data"].get("train_cache_rate", 0.5)
    val_cache_rate = config["data"].get("val_cache_rate", 0.0)

    train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=train_cache_rate, num_workers=4) if train_data else None
    val_ds   = CacheDataset(data=val_data,   transform=val_transforms, cache_rate=val_cache_rate, num_workers=4) if val_data else None
    # no need to cache the test set
    test_ds  = Dataset(data=test_data,  transform=val_transforms) if test_data else None
    # create a ds for the training set with validation transforms for final eval
    eval_train_ds = Dataset(data=train_data, transform=val_transforms) if train_data else None

    train_loader = DataLoader(train_ds, batch_size=config["data"]["batch_size"], shuffle=True, num_workers=config["data"]["workers"], pin_memory=True, persistent_workers=True) if train_ds else None
    val_loader   = DataLoader(val_ds,   batch_size=config["data"]["val_batch_size"], shuffle=False, num_workers=config["data"]["workers"], pin_memory=True, persistent_workers=True) if val_ds else None
    test_loader  = DataLoader(test_ds,  batch_size=config["data"]["val_batch_size"], shuffle=False, num_workers=config["data"]["workers"], pin_memory=True, persistent_workers=True) if test_ds else None
    eval_train_dataloader = DataLoader(eval_train_ds, batch_size=config["data"]["val_batch_size"], shuffle=False, num_workers=config["data"]["workers"], pin_memory=True, persistent_workers=True) if eval_train_ds else None

    model = create_model(config)

    # Setup output directory
    output_dir = os.path.join(config["output"]["path"], config["output"]["save_name"])
    if not os.path.exists(output_dir):
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    else:
        logger.info(f"Output directory already exists: {output_dir}")

    save_config(config, os.path.join(output_dir, 'config.yaml'))

    # make our model trainer
    trainer = ModelTrainer(model, device, config, output_dir)

    if not args.predict_only:
        trainer.train(train_loader, val_loader, disable_pbar=args.disable_progress_bar)

    # Select checkpoint based on configuration
    if args.predict_only:
        logger.info("Running in prediction mode, using the passed checkpoint if available")
        if config["model"].get("checkpoint_path"):
            best_checkpoint = config["model"]["checkpoint_path"]
            if not os.path.exists(best_checkpoint):
                logger.error(f"Checkpoint path {best_checkpoint} does not exist, exiting")
                return
        else:
            logger.error("predict_only mode requires checkpoint_path in config")
            return
    
    elif trainer.use_last_model or trainer.validation_mode in ["monitor_only", "none"]:
        # Use last epoch checkpoint
        best_checkpoint = os.path.join(trainer.checkpoint_dir, f"last_epoch_{trainer.epoch}.ckpt")
        logger.info(f"Using last epoch model: {best_checkpoint}")
    
    elif trainer.saved_checkpoints:
        # Use best validation checkpoint (early_stopping mode)
        best_checkpoint = trainer.saved_checkpoints[-1]
        logger.info(f"Using best validation model: {best_checkpoint}")
    
    else:
        logger.warning("No checkpoints found, using current model state")
        best_checkpoint = None

    logger.info(f"Selected checkpoint: {best_checkpoint}")

    # test the model
    if test_loader:
        test_results = trainer.test(test_loader, checkpoint_path=best_checkpoint, disable_pbar=args.disable_progress_bar)
        logger.info(f"test Results: {test_results}")
    else:
        logger.warning("no test loader provided, skipping testing")

    # final test on the train and validation sets
    if eval_train_dataloader:
        logger.info("Evaluating on training set with validation transforms...")
        train_results = trainer.test(eval_train_dataloader, checkpoint_path=best_checkpoint, disable_pbar=args.disable_progress_bar)
        logger.info(f"Train Results: {train_results}")
    else:
        logger.warning("No eval train dataloader provided, skipping evaluation on training set")
    
    if val_loader:
        logger.info("Evaluating on validation set...")
        val_results = trainer.test(val_loader, checkpoint_path=best_checkpoint, disable_pbar=args.disable_progress_bar)
        logger.info(f"Validation Results: {val_results}")
    else:
        logger.warning("No validation loader provided, skipping evaluation on validation set")

    # save a small results summary
    results_summary = {
        "best_checkpoint": best_checkpoint,
        "test_results": test_results if test_loader else None,
        "train_results": train_results if eval_train_dataloader else None,
        "val_results": val_results if val_loader else None
    }

    # save the results summary to a JSON file
    with open(os.path.join(output_dir, "results_summary.json"), "w") as f:
        json.dump(results_summary, f, indent=4)
    logger.info(f"Results summary saved to {os.path.join(output_dir, 'results_summary.json')}")


    # predictions for combination with other modalities
    if config["output"].get("prediction_dir", None) is not None:
        logger.info("Saving predictions...")
        
        pred_path = os.path.join(output_dir, config["output"]["prediction_dir"])

        os.makedirs(pred_path, exist_ok=True)
        
        logger.info(f"Saving predictions to {pred_path}")
        
        if eval_train_dataloader:
            preds = trainer.predict(eval_train_dataloader, checkpoint_path=best_checkpoint, disable_pbar=args.disable_progress_bar)
            preds.to_csv(os.path.join(pred_path, f"{config['output']['save_name']}_train_predictions.csv"), index=False)
        
        if val_loader:
            preds = trainer.predict(val_loader, checkpoint_path=best_checkpoint, disable_pbar=args.disable_progress_bar)
            preds.to_csv(os.path.join(pred_path, f"{config['output']['save_name']}_val_predictions.csv"), index=False)
        
        if test_loader:
            preds = trainer.predict(test_loader, checkpoint_path=best_checkpoint, disable_pbar=args.disable_progress_bar)
            preds.to_csv(os.path.join(pred_path, f"{config['output']['save_name']}_test_predictions.csv"), index=False)
        else:
            logger.warning("No test loader provided, skipping predictions on test set")
    else:
        logger.warning("No prediction directory specified in config, skipping predictions")
    
if __name__ == "__main__":
    main()