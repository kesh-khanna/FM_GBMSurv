"""
BrainIAC ViT-B backbone.
Pretrained with SimCLR on single-channel structural brain MRI (96^3 volumes).
Reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC11643205/
"""
import torch
import torch.nn as nn
from monai.networks.nets import ViT


def build_brainiac_vit() -> ViT:
    """Instantiate the ViT-B architecture matching BrainIAC pretraining config."""
    return ViT(
        in_channels=1,
        img_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        hidden_size=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12,
        classification=False,
        save_attn=False,
    )


def load_brainiac_weights(encoder: ViT, checkpoint_path: str) -> None:
    """Load BrainIAC weights into a bare MONAI ViT encoder.

    Handles two checkpoint formats automatically:
      - SimCLR pretrained (BrainIAC.ckpt): keys are 'backbone.<vit_key>'
      - Downstream fine-tuned (OS_ViT_BrainIAC.ckpt): keys are 'backbone.backbone.<vit_key>'
        (ViTBackboneNet wraps the ViT as self.backbone, adding an extra prefix layer)
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    # Try double prefix first (downstream checkpoint), then single prefix (SimCLR)
    for prefix in ("backbone.backbone.", "backbone."):
        new_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if new_sd:
            print(f"  Detected checkpoint format: '{prefix}*' prefix ({len(new_sd)} keys)")
            break
    else:
        found_prefixes = sorted(set(k.split(".")[0] for k in sd.keys()))
        raise ValueError(
            f"No 'backbone.' or 'backbone.backbone.' keys found in {checkpoint_path}. "
            f"Top-level prefixes: {found_prefixes}"
        )

    out = encoder.load_state_dict(new_sd, strict=False)
    print(f"BrainIAC weights loaded | missing={len(out.missing_keys)} unexpected={len(out.unexpected_keys)}")
    if out.unexpected_keys:
        print(f"  Unexpected (checkpoint has keys not in model): {out.unexpected_keys}")
    expected_missing = {k for k in out.missing_keys if "cross_attn" in k or "norm_cross_attn" in k}
    real_missing = set(out.missing_keys) - expected_missing
    if real_missing:
        print(f"  WARNING — unexpected missing backbone keys: {real_missing}")


def build_brainiac_projection_head(checkpoint_path: str) -> nn.Sequential:
    """Load the SimCLR projection head from a BrainIAC.ckpt checkpoint.

    Architecture (from checkpoint inspection):
        Linear(768 → 768) → BatchNorm1d(768) → ReLU → Linear(768 → 2048) → BatchNorm1d(2048)

    Output dim is 2048, matching the paper's reported feature dimensionality.
    Only valid for the SimCLR pretrained checkpoint (BrainIAC.ckpt); downstream
    fine-tuned checkpoints (e.g. OS_ViT_BrainIAC.ckpt) do not contain this head.
    """
    head = nn.Sequential(
        nn.Linear(768, 768, bias=False),
        nn.BatchNorm1d(768),
        nn.ReLU(inplace=True),
        nn.Linear(768, 2048, bias=False),
        nn.BatchNorm1d(2048),
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    # Checkpoint stores the head as projection_head.layers.<idx>.*; strip both prefixes
    # so keys match nn.Sequential's integer-indexed naming (<idx>.*).
    proj_sd = {
        k[len("projection_head.layers."):]: v
        for k, v in sd.items()
        if k.startswith("projection_head.layers.")
    }
    if not proj_sd:
        raise ValueError(
            f"No 'projection_head.layers.*' keys found in {checkpoint_path}. "
            "This checkpoint may not be a SimCLR pretrained BrainIAC.ckpt."
        )
    head.load_state_dict(proj_sd, strict=True)
    print(f"BrainIAC projection head loaded ({len(proj_sd)} keys) → output dim 2048")
    return head


def set_trainable_brainiac(
    encoder: ViT,
    frozen_blocks: int = 0,
    train_patch_embed: bool = True,
    train_norm: bool = True,
) -> None:
    """Control which parts of the ViT are frozen.

    ViT-B has 12 transformer blocks (blocks.0 ... blocks.11).
    frozen_blocks=0:  all unfrozen (full fine-tune).
    frozen_blocks=12: backbone fully frozen; only norm + patch_embed optionally unfrozen.

    Args:
        frozen_blocks:     Number of transformer blocks to freeze (from block 0 upward).
        train_patch_embed: Whether to keep patch_embedding trainable.
        train_norm:        Whether to keep the final layer norm trainable.
    """
    for param in encoder.parameters():
        param.requires_grad = False

    if train_patch_embed:
        for param in encoder.patch_embedding.parameters():
            param.requires_grad = True

    for i, block in enumerate(encoder.blocks):
        if i >= frozen_blocks:
            for param in block.parameters():
                param.requires_grad = True

    if train_norm and hasattr(encoder, "norm"):
        for param in encoder.norm.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total = sum(p.numel() for p in encoder.parameters())
    print(
        f"BrainIAC trainable: {trainable:,}/{total:,} params "
        f"({100 * trainable / total:.1f}%) — "
        f"frozen_blocks={frozen_blocks}, train_patch_embed={train_patch_embed}, train_norm={train_norm}"
    )
