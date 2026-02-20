"""
Author: Rakesh Khanna
"""
import argparse
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from monai.data import set_track_meta
import time
import os
from monai.data import Dataset, CacheDataset, DataLoader

from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.cindex import ConcordanceIndex
from tqdm import tqdm
import logging
from torch.amp import GradScaler
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch

from create_model import create_model
from transforms.transforms import custom_transform
from utils.torch_utils import set_seed, set_bn_eval

from utils.utils import check_censoring, load_config, save_config
from optimizers.create_optimizer import create_optimizer_scheduler

class ModelTrainer:
    def __init__(self, model, device, config, output_dir):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.output_dir = output_dir
        
        self.max_epochs = config["training"]["max_epochs"]
        self.head_lr = config["training"]["head_lr"]
        self.backbone_lr = config["training"]["backbone_lr"]
        self.weight_decay = config["training"]["reg_weight"]
        
        # validation settings
        self.evaluation_strategy = config["training"].get("evaluation_strategy", "last_epoch")
        self.patience = config["training"].get("patience", 10)
        
        # Validation modes: "early_stopping", "monitor_only", "none"
        assert self.evaluation_strategy in ["best_val_cindex", "last_epoch"], \
            f"evaluation_strategy must be one of: best_val_cindex, or last_epoch. Got: {self.evaluation_strategy}"

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
        self.score = -float('inf')

        self.patience_counter = 0
        self.global_step = 0
        self.epoch = 0

        # setup tensorboard
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))
        
        # create checkpoint directory
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training metrics
        self.training_auc = None
        self.training_c = None
        # Validation metrics
        self.val_auc = None
        self.val_c = None
        # Test metrics
        self.testing_auc = None
        self.testing_c = None
    
    def save_checkpoint(self, is_best=False, is_last=False):
        """
        save model checkpoint and relevant states if they are available
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'score': self.score,
            'global_step': self.global_step,
            'config': self.config,
            'scaler': self.scaler.state_dict() if hasattr(self, 'scaler') else None, 
        }
        
        if is_best:
            # denote in filename that this is the best model with epoch and validation C-Index
            checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{self.epoch}_val_cindex_{self.score:.3f}.ckpt')

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
        """Load model eval_checkpoint"""
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

        if "round" in checkpoint:
            logger.info(f"Round loaded from checkpoint: {checkpoint['round']}")

        self.score = checkpoint['score']
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Model loaded from: epoch {self.epoch}, global step {self.global_step}")

        return True
    
    def select_checkpoint_for_evaluation(self, predict_only=False, checkpoint_path=None):
        """
        Determine which checkpoint we want to use for the final evaluation
        """
        # if we are in predict only mode then 
        # Select eval_checkpoint based on configuration
        if predict_only:
            logger.info("Running in prediction mode, using the passed eval_checkpoint if available")
            if checkpoint_path:
                eval_checkpoint = checkpoint_path
                if not os.path.exists(eval_checkpoint):
                    logger.error(f"Checkpoint path {eval_checkpoint} does not exist, exiting")
                    return
            else:
                logger.error("predict_only mode requires checkpoint_path in config")
                return
        
        # in these cases we want to always eval with the last model
        elif self.evaluation_strategy == "last_epoch":
            eval_checkpoint = os.path.join(self.checkpoint_dir, f"last_epoch_{self.epoch}.ckpt")
            logger.info(f"Using last epoch model: {eval_checkpoint}")
        
        elif self.saved_checkpoints and self.evaluation_strategy == "best_val_cindex":
            eval_checkpoint = self.saved_checkpoints[-1]
            logger.info(f"Using best validation model: {eval_checkpoint}")
        
        else:
            logger.warning("No checkpoints found, using current model state")
            eval_checkpoint = None

        # logger.info(f"Selected {eval_checkpoint} as our checkpoint to be used for evaluation")
    
        return eval_checkpoint
    
    def train_epoch(self, train_loader, disable_pbar=False):
        """
        Train for one epoch.
        """
        self.model.train()

        # reset to be safe
        self.training_auc = Auc()
        self.training_c = ConcordanceIndex()

        # if we want to freeze the batchnorm then we have to do it at every epoch
        if self.config["model"].get("freeze_batchnorm", False):
            self.model.apply(set_bn_eval)

        total_loss = 0.0
        num_loss_computations = 0

        all_log_hz, all_time, all_events = [], [], []

        for batch_idx, batch in enumerate(tqdm(train_loader, desc='Training', disable=disable_pbar)):
            images = batch['image'].to(self.device, non_blocking=True)
            time = batch['label'].to(self.device, non_blocking=True)
            events = batch['event'].bool().to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.config["training"]["mixed_precision"]):
                log_hz = self.model(images)  

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
            # for GBM this is often not an issue
            if events.sum() == 0:
                if not disable_pbar:
                    tqdm.write(f"Batch {batch_idx+1}: no events, skipping update")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            # Backward + step
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()

            # log gradient norms periodically to track relative learning and monitor any spikes / make sure the gradscaler is working okay
            if batch_idx % 20 == 0:  # every 20 batches
                backbone_norm = 0
                for p in self.optimizer.param_groups[0]['params']:
                    if p.grad is not None:
                        backbone_norm += p.grad.data.norm(2).item() ** 2
                backbone_norm = backbone_norm ** 0.5
                
                head_norm = 0 # currnently also includes pooling params
                for p in self.optimizer.param_groups[1]['params']:
                    if p.grad is not None:
                        head_norm += p.grad.data.norm(2).item() ** 2
                head_norm = head_norm ** 0.5
                
                # Total
                total_norm = (backbone_norm ** 2 + head_norm ** 2) ** 0.5
                
                self.writer.add_scalar('gradients/backbone_norm', backbone_norm, self.global_step)
                self.writer.add_scalar('gradients/head_norm', head_norm, self.global_step)
                self.writer.add_scalar('gradients/total_norm', total_norm, self.global_step)
                
                # ratio for the relative learning
                if backbone_norm > 0:
                    self.writer.add_scalar('gradients/head_to_backbone_ratio', head_norm / backbone_norm, self.global_step)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_loss_computations += 1
            self.global_step += 1

        avg_loss = total_loss / num_loss_computations if num_loss_computations > 0 else float('inf')

        # Epoch-level metrics
        if all_log_hz:
            all_log_hz = torch.cat(all_log_hz, dim=0)
            all_time = torch.cat(all_time, dim=0)
            all_events = torch.cat(all_events, dim=0).bool()

            new_time = self.new_time
            train_auc = self.training_auc(all_log_hz, all_events, all_time, new_time=new_time)
            train_c = self.training_c(all_log_hz, all_events, all_time)

        else:
            train_auc = torch.Tensor(0.0)
            train_c = torch.Tensor(0.0)

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
        
        all_log_hz, all_time, all_events = [], [], []
        
        logger.info("collecting predictions for validation...")

        self.val_auc = Auc()
        self.val_c = ConcordanceIndex()    

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validation', disable=disable_pbar)):
                images = batch['image'].to(self.device, non_blocking=True)
                time = batch['label'].to(self.device, non_blocking=True)
                events = batch['event'].bool().to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=self.config["training"]["mixed_precision"]):
                    log_hz = self.model(images)
                
                all_log_hz.append(log_hz.cpu())
                all_time.append(time.cpu())
                all_events.append(events.cpu())
    
        if all_log_hz:
            # cat all predictions and targets
            all_log_hz = torch.cat(all_log_hz, dim=0)
            all_time = torch.cat(all_time, dim=0)
            all_events = torch.cat(all_events, dim=0).bool()
            new_time = self.new_time
            
            # get the loss on the entire validation set 
            # again dont use amp since our loss doesnt accept half precision. Use float 64 instead to be safe
            # NOTE: due to the nature of our loss function (nll of the coxph model) the loss on the full validaiton 
            # set will not be directly comparable to the average per batch loss during training
            # please keep this in mind when you are looking at the absolute validation loss values compared to training
            # TODO: explore normalizations that will allow more direct comparisons
            with torch.amp.autocast("cuda", enabled=False):
                all_log_hz = all_log_hz.double()
                val_loss = self.loss_module(all_log_hz, event=all_events, time=all_time, reduction="mean").item()
            
            # calculate the metrics
            val_auc = self.val_auc(all_log_hz, all_events, all_time, new_time=new_time)
            val_c = self.val_c(all_log_hz, all_events, all_time)    

        else:
            val_loss = torch.Tensor(float('inf'))
            val_auc = torch.Tensor(0.0)
            val_c = torch.Tensor(0.0)
        
        # Log metrics
        self.writer.add_scalar('Loss/Validation', val_loss, self.epoch)
        self.writer.add_scalar('AUC/Validation', val_auc.item(), self.epoch)
        self.writer.add_scalar('C-Index/Validation', val_c.item(), self.epoch)
        
        logger.info(f"Val - Loss: {val_loss:.4f}, AUC: {val_auc.item():.4f}, C-Index: {val_c.item():.4f}")
        
        return val_loss, val_auc.item(), val_c.item()
    
    def train(self, train_loader, val_loader=None, disable_pbar=False):
        """
        Full training loop with validation and early stopping
        Train for max_epochs or until early stopping is triggered (if early stopping is requested in the config)
        """
        print("\n", "-"*80)
        logger.info(f"Starting training for {self.max_epochs} epochs")
        logger.info(f"Evaluation strategy: {self.evaluation_strategy}")
        logger.info(f"using device: {self.device}")
        logger.info(f"Training batch size: {self.config['data']['batch_size']}")
        if val_loader is not None:
            logger.info(f"Validation batch size: {self.config['data']['val_batch_size']}")
        logger.info(f"Weight Decay: {self.weight_decay}")
        logger.info(f"Encoder LR: {self.backbone_lr}")
        logger.info(f"Head LR: {self.head_lr}")
        print("-"*80, '\n')

        start_time = time.time()
        
        for epoch in range(self.max_epochs):
            self.epoch = epoch

            # Training
            self.train_epoch(train_loader, disable_pbar=disable_pbar)
            
            if self.scheduler is not None:
                self.scheduler.step()

            # Validation
            if val_loader is not None:
                val_loss, val_auc, val_c = self.validate_full_dataset(val_loader, disable_pbar=disable_pbar)
                
                monitor = val_c
                is_best = monitor > self.score
                
                if is_best:
                    self.score = max(monitor, self.score)
                    logger.info(f"New best validation C-Index: {val_c:.4f}")
                    
                    # Only save as "best" if using early stopping
                    if self.evaluation_strategy == "best_val_cindex":
                        self.patience_counter = 0
                        self.save_checkpoint(is_best=True)
                
                # Early stopping 
                if self.evaluation_strategy == "best_val_cindex":
                    if not is_best:
                        self.patience_counter += 1
                    
                    if self.patience_counter >= self.patience:
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        self.save_checkpoint(is_best=False, is_last=True)
                        break
        
            
            if (epoch + 1) % self.checkpoint_frequency == 0 or (epoch + 1) == self.max_epochs:
                self.save_checkpoint(is_best=False)
            

        training_time = (time.time() - start_time) / 60
        logger.info(f"Training completed in {training_time:.2f} minutes")
        
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / 1e9
            logger.info(f"Peak GPU memory usage: {max_memory:.2f} GB")
        
        self.writer.close()
        
    def eval_predict(self, data_loader, checkpoint_path=None, disable_pbar=False, dataset_name="Test"):
        """
        Evaluate for testing and store the final predictions for potential saving
        """
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path=checkpoint_path)

        self.model.eval()

        all_log_hz, all_time, all_events, all_patient_ids = [], [], [], []

        # reset the metrics
        self.testing_auc = Auc()
        self.testing_c = ConcordanceIndex()   

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating", disable=disable_pbar)):
                images = batch['image'].to(self.device)
                time = batch['label'].to(self.device)
                events = batch['event'].bool().to(self.device)
                patient_id = batch['patient_id']

                with torch.amp.autocast("cuda", enabled=self.config["training"]["mixed_precision"]):
                    log_hz = self.model(images)

                all_log_hz.append(log_hz.cpu())
                all_time.append(time.cpu())
                all_events.append(events.cpu())
                all_patient_ids.extend(patient_id)

        results = {}
        pred_df = None

        if all_log_hz:
            # concat the tensors
            all_log_hz_tensor = torch.cat(all_log_hz, dim=0).float()
            all_time_tensor = torch.cat(all_time, dim=0).float()
            all_events_tensor = torch.cat(all_events, dim=0).bool()

            # compute metrics
            test_auc = self.testing_auc(all_log_hz_tensor, all_events_tensor, all_time_tensor, new_time=self.new_time.float())
            test_auc_ci = self.testing_auc.confidence_interval(method="bootstrap")
            test_c = self.testing_c(all_log_hz_tensor, all_events_tensor, all_time_tensor)
            test_c_ci = self.testing_c.confidence_interval(method="bootstrap")

            results = {
                f'{dataset_name}_auc': test_auc.item(),
                f'{dataset_name}_c_index': test_c.item(),
                f'{dataset_name}_auc_ci': test_auc_ci.tolist(),
                f'{dataset_name}_c_index_ci': test_c_ci.tolist()
            }
            
            logger.info(f"{dataset_name} - AUC: {test_auc.item():.4f}, C-Index: {test_c:.4f}")
            logger.info(f"{dataset_name} - AUC CI: {test_auc_ci.tolist()}, C-Index CI: {test_c_ci.tolist()}")

            pred_df = pd.DataFrame({
                'patient_id': all_patient_ids,
                'log_hz': all_log_hz_tensor.numpy().flatten(),
                'time': all_time_tensor.numpy().flatten(),
                'event': all_events_tensor.numpy().astype(bool).flatten()
            })

        else:
            logger.warning("No data processed")
            results = {f'{dataset_name}_auc': 0.0, f'{dataset_name}_c_index': 0.0, f'{dataset_name}_auc_ci': None, f'{dataset_name}_c_index_ci': None}
        
        return results, pred_df

def create_parser():
    parser = argparse.ArgumentParser(description="FL_BrainSurViT Driver")
    parser.add_argument("--config_file", required=True, type=str, help="Path to YAML config for the model and training")
    parser.add_argument("--predict_only", action="store_true", help="Only make predictions")
    parser.add_argument("--disable_progress_bar", action="store_true", help="Disable progress bar for training and validation")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    config = load_config(args.config_file)
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_track_meta(True)
    
    logger.info(f"config loaded from: {args.config_file}")
    print("-"*80, "\n")

    # load in the json file
    with open(config["data"]["json_file"], "r") as f:
        json_data = json.load(f)

    train_data = json_data.get("train", None)
    val_data = json_data.get("validation", None)
    test_data = json_data.get("test", None)

    # check the amount of censoring in each split
    check_censoring(train_data, "Training")
    check_censoring(val_data, "Validation")
    check_censoring(test_data, "Testing")

    train_transforms, val_transforms = custom_transform(config)

    if args.predict_only:
        # no need to cache if we are only predicting
        train_cache_rate = 0.0
        val_cache_rate = 0.0
    else:
        train_cache_rate = config["data"].get("train_cache_rate", 0.0)
        val_cache_rate = config["data"].get("val_cache_rate", 0.0)

    # CacheDataset will cache up to the first Randomizable transformation, in our context this will mainly be the loading,
    # the orientation / spacing transforms, and normalizations
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

    model = create_model(config, args.predict_only)

    # Setup output directory
    output_dir = os.path.join(config["output"]["path"], config["output"]["save_name"])
    if not os.path.exists(output_dir):
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    else:
        logger.info(f"Output directory already exists: {output_dir}")


    save_config(config, os.path.join(output_dir, 'config.yaml')) #TODO make sure that we are saving the config after all defaults have been added in.

    # make our model self
    trainer = ModelTrainer(model, device, config, output_dir)

    if not args.predict_only:
        trainer.train(train_loader, val_loader, disable_pbar=args.disable_progress_bar)
    
    print("\n", "-"*80)
    logger.info("Starting the final evaluation and saving")
    print("-"*80, "\n")
    eval_checkpoint = trainer.select_checkpoint_for_evaluation(predict_only=args.predict_only, checkpoint_path=config["model"].get("checkpoint_path", None))
    
    # start a small results summary
    results_summary = {
        "eval_checkpoint": eval_checkpoint
    }

    # test the model
    if test_loader:
        test_results, test_preds = trainer.eval_predict(
            test_loader, 
            checkpoint_path=eval_checkpoint, 
            disable_pbar=args.disable_progress_bar,
            dataset_name = "Test"
        )
        results_summary["test_results"] = test_results
        print("-"*80, "\n")
    else:
        logger.warning("No test data provided, skipping evaluation on \"test set\"")
        print("-"*80, "\n")
        test_results, test_preds = None, None

    # final test on the train and validation sets
    if eval_train_dataloader:
        logger.info("Evaluating on training set with validation transforms...")
        train_results, train_preds = trainer.eval_predict(
            eval_train_dataloader, 
            checkpoint_path=eval_checkpoint, 
            disable_pbar=args.disable_progress_bar,
            dataset_name="Training"
        )
        results_summary["train_results"] = train_results
        print("-"*80, "\n")
    else:
        logger.warning("No eval train data provided, skipping evaluation on training set")
        print("-"*80, "\n")
        train_results, train_preds = None, None
    
    if val_loader:
        logger.info("Evaluating on validation set...")
        val_results, val_preds = trainer.eval_predict(
            val_loader, 
            checkpoint_path=eval_checkpoint, 
            disable_pbar=args.disable_progress_bar,
            dataset_name="Validation"
        )
        results_summary["val_results"] = val_results
        print("-"*80, "\n")
    else:
        logger.warning("No validation data provided, skipping evaluation on validation set")
        print("-"*80, "\n")

        val_results, val_preds = None, None

    # save the results summary to a JSON file
    with open(os.path.join(output_dir, "results_summary.json"), "w") as f:
        json.dump(results_summary, f, indent=4)
    logger.info(f"Results summary saved to {os.path.join(output_dir, 'results_summary.json')}")
    print("-"*80, "\n")

    if config["output"].get("prediction_dir", None) is not None:
        logger.info("Saving predictions...")
        pred_path = os.path.join(output_dir, config["output"]["prediction_dir"])
        os.makedirs(pred_path, exist_ok=True)

        if train_preds is not None:
            train_preds.to_csv(os.path.join(pred_path, f"{config['output']['save_name']}_train_predictions.csv"), index=False)
            logger.info("Saved training set predictions")
        if val_preds is not None:
            val_preds.to_csv(os.path.join(pred_path, f"{config['output']['save_name']}_val_predictions.csv"), index=False)
            logger.info("Saved validation set predictions")
        if test_preds is not None:
            test_preds.to_csv(os.path.join(pred_path, f"{config['output']['save_name']}_test_predictions.csv"), index=False)
            logger.info("Saved test set predictions")

    else:
        logger.warning("No prediction directory specified in config, skipping predictions")
    
if __name__ == "__main__":
    main()