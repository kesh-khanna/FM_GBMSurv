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

from tqdm import tqdm
import logging
from torch.amp import GradScaler
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from create_model import create_model
from transforms.transforms import custom_transform
from utils.torch_utils import (set_seed, set_bn_eval, cox_loss_f64,
                               compute_survival_metrics, compute_group_grad_norms,
                               load_model_weights)
from utils.utils import check_censoring, load_config, save_config, plot_training_curves, save_prediction_csvs
from utils.data_utils import load_split_data, build_loaders
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

        self.scaler = GradScaler(enabled=self.config["training"]["mixed_precision"])

        # how often to save a standard checkpoint
        self.checkpoint_frequency = config["training"]["checkpoint_frequency"]
        self.save_top_k = config["training"].get("save_top_k", None)  # none means save all
        self.saved_checkpoints = []

        self.optimizer, self.scheduler = create_optimizer_scheduler(self.model, config)

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

        # Loss and C-Index history for matplotlib plot
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_cindex: list[float] = []
        self.val_cindex: list[float] = []

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

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
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

    def load_weights_for_eval(self, checkpoint_path):
        """Load only the model weights for evaluation; raises instead of returning False
        so a bad path can never silently fall through to an untrained model."""
        checkpoint = load_model_weights(self.model, checkpoint_path, self.device)
        logger.info(f"Evaluation weights loaded from {checkpoint_path} (epoch {checkpoint.get('epoch')})")

    def select_checkpoint_for_evaluation(self, predict_only=False, checkpoint_path=None):
        """
        Determine which checkpoint we want to use for the final evaluation
        """
        # if we are in predict only mode then
        # Select eval_checkpoint based on configuration
        if predict_only:
            logger.info("Running in prediction mode, using the passed eval_checkpoint")
            if not checkpoint_path:
                raise ValueError(
                    "predict_only mode requires a checkpoint: set model.checkpoint_path in the "
                    "config or pass --checkpoint. Refusing to evaluate an untrained model."
                )
            eval_checkpoint = checkpoint_path
            if not os.path.exists(eval_checkpoint):
                raise FileNotFoundError(f"Checkpoint path {eval_checkpoint} does not exist")

        # in these cases we want to always eval with the last model
        elif self.evaluation_strategy == "last_epoch":
            eval_checkpoint = os.path.join(self.checkpoint_dir, f"last_epoch_{self.epoch}.ckpt")
            if not os.path.exists(eval_checkpoint):
                raise FileNotFoundError(
                    f"Expected last-epoch checkpoint at {eval_checkpoint} but it does not exist"
                )
            logger.info(f"Using last epoch model: {eval_checkpoint}")

        elif self.saved_checkpoints and self.evaluation_strategy == "best_val_cindex":
            eval_checkpoint = self.saved_checkpoints[-1]
            logger.info(f"Using best validation model: {eval_checkpoint}")

        else:
            raise RuntimeError(
                f"No checkpoint available for evaluation_strategy='{self.evaluation_strategy}'. "
                "best_val_cindex requires a validation split so best-val checkpoints are saved; "
                "use evaluation_strategy='last_epoch' or provide validation data."
            )

        return eval_checkpoint

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
            images = batch['image'].to(self.device, non_blocking=True)
            time = batch['label'].to(self.device, non_blocking=True)
            events = batch['event'].bool().to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.config["training"]["mixed_precision"]):
                log_hz = self.model(images)

            # Save detached copies for epoch metrics.
            # NOTE: this happens before the zero-event skip below, so batches that
            # contribute no gradient still contribute to the epoch-level metrics.
            all_log_hz.append(log_hz.detach().cpu())
            all_time.append(time.detach().cpu())
            all_events.append(events.detach().cpu())

            loss = cox_loss_f64(log_hz, events, time)

            # skip if no events in this mini-batch
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
                # group 0 = backbone, group 1 = head (currently also includes pooling params)
                norms = compute_group_grad_norms(self.optimizer)
                backbone_norm, head_norm = norms[0], norms[1]
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
            metrics = compute_survival_metrics(
                torch.cat(all_log_hz, dim=0),
                torch.cat(all_events, dim=0).bool(),
                torch.cat(all_time, dim=0),
                self.new_time,
            )
            train_auc, train_c = metrics["auc"], metrics["c_index"]
        else:
            train_auc, train_c = 0.0, 0.0

        # Log metrics
        self.writer.add_scalar('Loss/Train', avg_loss, self.epoch)
        self.writer.add_scalar('AUC/Train', train_auc, self.epoch)
        self.writer.add_scalar('C-Index/Train', train_c, self.epoch)
        for i, g in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'LR/group_{i}', g['lr'], self.epoch)

        logger.info(f"Train - Avg Loss: {avg_loss:.4f}, AUC: {train_auc:.4f}, C-Index: {train_c:.4f}")
        return avg_loss, train_c


    def validate_full_dataset(self, val_loader, disable_pbar=False):
        """
        Validate on entire dataset at once for survival prediction.
        This accumulates all predictions and targets before computing loss.
        """
        self.model.eval()

        all_log_hz, all_time, all_events = [], [], []

        logger.info("collecting predictions for validation...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validation', disable=disable_pbar)):
                images = batch['image'].to(self.device, non_blocking=True)
                time = batch['label'].to(self.device, non_blocking=True)
                events = batch['event'].bool().to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=self.config["training"]["mixed_precision"]):
                    out = self.model(images)

                all_log_hz.append(out.cpu())
                all_time.append(time.cpu())
                all_events.append(events.cpu())

        if all_log_hz:
            # NOTE: Cox NLL on full val set is not directly comparable to per-batch train loss
            metrics = compute_survival_metrics(
                torch.cat(all_log_hz, dim=0),
                torch.cat(all_events, dim=0).bool(),
                torch.cat(all_time, dim=0),
                self.new_time,
                with_loss=True,
            )
            val_loss, val_auc, val_c = metrics["loss"], metrics["auc"], metrics["c_index"]
        else:
            val_loss, val_auc, val_c = float('inf'), 0.0, 0.0

        # Log metrics
        self.writer.add_scalar('Loss/Validation', val_loss, self.epoch)
        self.writer.add_scalar('AUC/Validation', val_auc, self.epoch)
        self.writer.add_scalar('C-Index/Validation', val_c, self.epoch)

        logger.info(f"Val - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, C-Index: {val_c:.4f}")

        return val_loss, val_auc, val_c

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
            train_loss, train_c = self.train_epoch(train_loader, disable_pbar=disable_pbar)
            self.train_losses.append(train_loss)
            self.train_cindex.append(train_c)

            if self.scheduler is not None:
                self.scheduler.step()

            # Validation
            if val_loader is not None:
                val_loss, val_auc, val_c = self.validate_full_dataset(val_loader, disable_pbar=disable_pbar)
                self.val_losses.append(val_loss)
                self.val_cindex.append(val_c)

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

        plot_training_curves(
            self.train_losses, self.val_losses, self.train_cindex, self.val_cindex,
            self.output_dir, title=f"Training curves — {self.config['output']['save_name']}",
        )
        self.writer.close()

    def eval_predict(self, data_loader, checkpoint_path=None, disable_pbar=False,
                     dataset_name="Test"):
        """
        Evaluate for testing and store the final predictions for potential saving.
        """
        if checkpoint_path:
            self.load_weights_for_eval(checkpoint_path)

        self.model.eval()

        all_log_hz, all_time, all_events, all_patient_ids = [], [], [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating", disable=disable_pbar)):
                images = batch['image'].to(self.device)
                time = batch['label'].to(self.device)
                events = batch['event'].bool().to(self.device)
                patient_id = batch['patient_id']

                with torch.amp.autocast("cuda", enabled=self.config["training"]["mixed_precision"]):
                    out = self.model(images)

                all_log_hz.append(out.cpu())
                all_time.append(time.cpu())
                all_events.append(events.cpu())
                all_patient_ids.extend(patient_id)

        results = {}
        pred_df = None

        if all_log_hz:
            all_log_hz_tensor = torch.cat(all_log_hz, dim=0).float()
            all_time_tensor = torch.cat(all_time, dim=0).float()
            all_events_tensor = torch.cat(all_events, dim=0).bool()

            metrics = compute_survival_metrics(
                all_log_hz_tensor, all_events_tensor, all_time_tensor,
                self.new_time.float(), with_ci=True,
            )

            results = {
                f'{dataset_name}_auc': metrics['auc'],
                f'{dataset_name}_c_index': metrics['c_index'],
                f'{dataset_name}_auc_ci': metrics['auc_ci'],
                f'{dataset_name}_c_index_ci': metrics['c_index_ci']
            }

            logger.info(f"{dataset_name} - AUC: {metrics['auc']:.4f}, C-Index: {metrics['c_index']:.4f}")
            logger.info(f"{dataset_name} - AUC CI: {metrics['auc_ci']}, C-Index CI: {metrics['c_index_ci']}")

            risk_col = 'log_hz'  # name preserved for CSV compatibility
            pred_df = pd.DataFrame({
                'patient_id': all_patient_ids,
                risk_col: all_log_hz_tensor.numpy().flatten(),
                'time': all_time_tensor.numpy().flatten(),
                'event': all_events_tensor.numpy().astype(bool).flatten()
            })

        else:
            logger.warning("No data processed")
            results = {f'{dataset_name}_auc': 0.0, f'{dataset_name}_c_index': 0.0,
                       f'{dataset_name}_auc_ci': None, f'{dataset_name}_c_index_ci': None}

        return results, pred_df

def create_parser():
    parser = argparse.ArgumentParser(description="FL_BrainSurViT Driver")
    parser.add_argument("--config_file", required=True, type=str, help="Path to YAML config for the model and training")
    parser.add_argument("--predict_only", action="store_true", help="Only make predictions")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint to evaluate; overrides model.checkpoint_path from the config")
    parser.add_argument("--disable_progress_bar", action="store_true", help="Disable progress bar for training and validation")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    config = load_config(args.config_file)
    if args.checkpoint:
        logger.info(f"Overriding model.checkpoint_path with --checkpoint {args.checkpoint}")
        config["model"]["checkpoint_path"] = args.checkpoint
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_track_meta(True)

    logger.info(f"config loaded from: {args.config_file}")
    print("-"*80, "\n")

    train_data, val_data, test_data = load_split_data(config["data"]["json_file"])

    # check the amount of censoring in each split
    check_censoring(train_data, "Training")
    check_censoring(val_data, "Validation")
    check_censoring(test_data, "Testing")

    train_transforms, val_transforms = custom_transform(config)

    train_loader, val_loader, test_loader, eval_train_dataloader = build_loaders(
        config, train_data, val_data, test_data, train_transforms, val_transforms,
        predict_only=args.predict_only,
    )

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
    # load the weights once here; eval_predict below is then called without a checkpoint path
    trainer.load_weights_for_eval(eval_checkpoint)

    # start a small results summary
    results_summary = {
        "eval_checkpoint": eval_checkpoint
    }

    pred_path = None
    if config["output"].get("prediction_dir", None) is not None:
        pred_path = os.path.join(output_dir, config["output"]["prediction_dir"])
        os.makedirs(pred_path, exist_ok=True)

    # test the model
    if test_loader:
        test_results, test_preds = trainer.eval_predict(
            test_loader,
            disable_pbar=args.disable_progress_bar,
            dataset_name="Test",
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
            disable_pbar=args.disable_progress_bar,
            dataset_name="Training",
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
            disable_pbar=args.disable_progress_bar,
            dataset_name="Validation",
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

    if pred_path is not None:
        logger.info("Saving predictions...")
        save_prediction_csvs(
            {"train": train_preds, "val": val_preds, "test": test_preds},
            pred_path, config["output"]["save_name"],
        )
    else:
        logger.warning("No prediction directory specified in config, skipping predictions")

if __name__ == "__main__":
    main()
