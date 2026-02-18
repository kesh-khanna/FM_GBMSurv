import torch
from typing import Tuple, Dict, Any
from torch.optim import Optimizer
from lr_scheduler import LinearWarmupCosineAnnealingLR

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

    if hasattr(model, "get_param_groups") and callable(model.get_param_groups):
        groups = model.get_param_groups()
        encoder_params = [p for p in groups.get("backbone", []) if p.requires_grad]
        head_params = [p for p in groups.get("head", []) if p.requires_grad]
    else:
        raise AttributeError(f"{model.__class__.__name__} must implement get_param_groups()")

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
