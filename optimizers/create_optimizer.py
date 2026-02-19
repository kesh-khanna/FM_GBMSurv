import torch
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from classifiers.survival_models import BasePredictionModel

def create_optimizer_scheduler(model: BasePredictionModel, config):
    """
    Create optimizer + scheduler with two learning rates:
      - backbone_lr: model.embedder.encoder (i.e. UniFormer)
      - head_lr: everything else (pooling + survival head)
    Only includes params with requires_grad=True.
    """

    wd = float(config["training"]["reg_weight"])
    backbone_lr = float(config["training"].get("backbone_lr")) # TODO make sure these 2 are required pieces in the config
    head_lr = float(config["training"].get("head_lr"))
    optim_name = config["training"]["optim_name"].lower()

    groups = model.get_param_groups()

    # only pass the ones that are trainable
    encoder_params = [p for p in groups.get("backbone", []) if p.requires_grad]

    # group the pooling with the head for now, later can add its own lr if needed
    pooling_params = [p for p in groups.get("pooling", []) if p.requires_grad]
    head_params = [p for p in groups.get("head", []) if p.requires_grad]
    all_head_params = pooling_params + head_params

    print(f"Using two LRs: backbone_lr={backbone_lr} head_lr={head_lr} wd={wd} optim={optim_name}")
    print(f"Trainable params: backbone={sum(p.numel() for p in encoder_params):,}, "
          f"head={sum(p.numel() for p in all_head_params):,} "
          f"(pooling={sum(p.numel() for p in pooling_params):,} + mlp={sum(p.numel() for p in head_params):,})")

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
