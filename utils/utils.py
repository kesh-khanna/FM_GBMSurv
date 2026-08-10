import os
import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def check_censoring(data, split_name):
    """
    check the censoring rate of the dataset
    """
    if data:
        items = data.values() if isinstance(data, dict) else data
        num_events = sum([int(d['event']) for d in items])
        num_censored = len(data) - num_events
        censoring_rate = num_censored / len(data)
        print(f"{split_name} - Total: {len(data)}, Events: {num_events}, Censored: {num_censored}, Censoring Rate: {censoring_rate:.2f}")
    else:
        print(f"{split_name} data is empty or not provided.")


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


def save_prediction_csvs(preds_by_split, pred_path, save_name):
    """save prediction DataFrames as {save_name}_{split}_predictions.csv, skipping missing splits"""
    for split, df in preds_by_split.items():
        if df is not None:
            df.to_csv(os.path.join(pred_path, f"{save_name}_{split}_predictions.csv"), index=False)
            logger.info(f"Saved {split} set predictions")


def plot_training_curves(train_losses, val_losses, train_cindex, val_cindex, output_dir, title):
    """Save a matplotlib figure with separate panels for train loss, val loss, and C-Index."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_val = bool(val_losses)
    train_epochs = list(range(len(train_losses)))
    val_epochs   = list(range(len(val_losses)))

    n_panels = 3 if has_val else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4))
    fig.subplots_adjust(wspace=0.38)

    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        plt.style.use("ggplot")

    def style_ax(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Epoch", fontsize=11)

    # --- Panel 1: Train Loss ---
    ax = axes[0]
    ax.plot(train_epochs, train_losses, color="#4C72B0", linewidth=1.8)
    ax.set_ylabel("Loss (Cox NLL, per batch)", fontsize=11)
    ax.set_title("Train Loss", fontsize=12, fontweight="bold")
    style_ax(ax)

    panel_idx = 1

    # --- Panel 2 (optional): Val Loss ---
    if has_val:
        ax = axes[panel_idx]
        ax.plot(val_epochs, val_losses, color="#DD8452", linewidth=1.8)
        ax.set_ylabel("Loss (Cox NLL, full val set)", fontsize=11)
        ax.set_title("Validation Loss", fontsize=12, fontweight="bold")
        style_ax(ax)
        panel_idx += 1

    # --- Final panel: C-Index ---
    ax = axes[panel_idx]
    ax.plot(train_epochs, train_cindex, label="Train", color="#4C72B0", linewidth=1.8)
    if has_val:
        ax.plot(val_epochs, val_cindex, label="Validation", color="#DD8452", linewidth=1.8)
        ax.legend(fontsize=10)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("C-Index", fontsize=11)
    ax.set_title("Concordance Index", fontsize=12, fontweight="bold")
    style_ax(ax)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)

    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"training_curves.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        logger.info(f"Training curves saved: {path}")

    plt.close(fig)
