"""
Author: Rakesh Khanna
"""

from backbones.uniformer import UniFormer, set_trainable_uniformer
from backbones.swin_encoder import SwinTransformer, set_trainable_swin
from backbones.brainiac_encoder import build_brainiac_vit, load_brainiac_weights, set_trainable_brainiac, build_brainiac_projection_head
from embedders.feature_extractors import (
    UniformerEmbedder, MultiScaleUniformerEmbedder,
    SwinViTEmbedder, TriadSwinViTEmbedder,
    BrainIACEmbedder, MultiScaleBrainIACEmbedder,
)
from classifiers.survival_models import DeepSurvNet

from typing import Dict, Any
import torch.nn as nn
import os
import torch


def _wrap_head(embedder, embedding_dim: int, config: Dict[str, Any]) -> nn.Module:
    """Wrap embedder in a DeepSurvNet (Cox proportional-hazards) head."""
    return DeepSurvNet(
        embedder=embedder,
        embedding_dim=embedding_dim,
        hidden_dims=config["model"].get("hidden_dims", [256]),
        return_embeddings=config["model"].get("return_embeddings", False),
    )


def create_model(config: Dict[str, Any], predict_only=False) -> nn.Module:
        """
        Create a model based on yaml config.
        Must have 'model.type' specifying the model type.
        Currently supported types: 'brainmvp', 'brainseg'
        Coming Soon: "brainiac"
        """
        model_type = config["model"].get("type", "brainmvp").lower()

        if model_type == "brainmvp":
            return create_model_brainmvp(config, predict_only)
        elif model_type == "brainseg":
            return create_model_brainseg(config, predict_only)
        elif model_type == "triadswb":
            return create_model_triadswb(config, predict_only)
        elif model_type == "brainiac":
            return create_model_brainiac(config, predict_only)
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Supported types: 'brainmvp', 'brainseg', 'triadswb', 'brainiac'"
            )

def create_model_brainmvp(config, predict_only):
    """
    create the encoder and wrap the encoder in the embedding model. could change to pass configs individually
    """
    # currently defaulted to match the weights of the BrainMVP, can be overwritten in the config if needed
    depths = config["model"].get("depths", [3, 4, 8, 3])
    encoder = UniFormer(
        depth=depths,
        img_size=config["model"]["img_size"],
        in_chans=config["model"]["in_chans"],
        num_classes=1,
        drop_rate=config["model"].get("drop_rate", 0.0),
        attn_drop_rate=config["model"].get("attn_drop_rate", 0.0),
        drop_path_rate=config["model"].get("drop_path_rate", 0.0),
    )

    if os.path.exists(config["model"]["pretrained_weights"]) and config["model"].get("use_pretrained_weights", False) and not predict_only:
        print("\n", "-"*80)
        print(f"Loading pretrained weights from {config['model']['pretrained_weights']}")
        # load in the weights
        weights = torch.load(config["model"]["pretrained_weights"], map_location="cpu")

        state_dict = {}
        for key in weights['state_dict'].keys():
            new_key = key.replace('module.', '').replace('uniformer.', '').replace('encoder.', '')
            state_dict[new_key] = weights['state_dict'][key]
        
        # duplicate the patch embedding 1 layer to accomadate k input channels insted of the 1 in the pretrained model
        # can change to full single modalities passes if alternative fusion methods are desired
        if config["model"]["in_chans"] != 1:
            print(f"Duplicating patch embedding weights to accomodate {config['model']['in_chans']} input channels")
            old_weight = state_dict['patch_embed1.proj.weight']  # [out_channels, in_channels, k, k, k]
            new_weight = old_weight.repeat(1, config["model"]["in_chans"], 1, 1, 1) / config["model"]["in_chans"]
            state_dict['patch_embed1.proj.weight'] = new_weight
        
        out = encoder.load_state_dict(state_dict, strict=False)
        print(f"Missing Keys: {out.missing_keys}")
        print(f"Unexpected Keys: {out.unexpected_keys}")
        print(f"Length of Missing Keys: {len(out.missing_keys)}")
        print(f"Length of Unexpected Keys: {len(out.unexpected_keys)}")

    elif predict_only:
        print("Created model for prediction...")

    else:
        print("Training model from scratch")
    
    # freeze portions of the encoder if needed
    # by default only the final stage and norm layers are trainable
    set_trainable_uniformer(
        encoder,
        train_patch_embed1=config["model"].get("train_patch_embed1", True),
        train_stage1=config["model"].get("train_stage1", True),
        train_stage2=config["model"].get("train_stage2", True),
        train_stage3=config["model"].get("train_stage3", True),
        train_stage4=config["model"].get("train_stage4", True),
        train_final_norm=config["model"].get("train_final_norm", True),
        train_all_layernorm=config["model"].get("train_all_layernorm", True),
        train_all_batchnorm=config["model"].get("train_all_batchnorm", True),
    )

    embedding_mode = config["model"].get("embedding_mode", "stage4")
    if embedding_mode == "multiscale":
        feat_dim = 1024  # 64+128+320+512 across UniFormer stages 1-4
        brain_embedder = MultiScaleUniformerEmbedder(encoder=encoder, feat_dim=feat_dim)
    else:
        feat_dim = 512
        brain_embedder = UniformerEmbedder(
            encoder=encoder,
            stage_idx=4,
            pooling=config["model"].get("pooling_method", "gap"),
            feat_dim=feat_dim,
        )

    return _wrap_head(brain_embedder, feat_dim, config)

from monai.utils import ensure_tuple_rep

def create_model_brainseg(config, predict_only):
    # some hardcoded values to match the weights, can move to configs if you want to change them
    spatial_dims = 3 
    img_size = ensure_tuple_rep(config["model"]["img_size"], spatial_dims)
    patch_sizes = ensure_tuple_rep(2, spatial_dims)
    window_size = ensure_tuple_rep(7, spatial_dims)
    num_heads = [3, 6, 12, 24]
    feature_size = 48

    # some checks in case we decide to move the hardcoded args to the config for flexibillity
    if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")
    if not (0 <= config["model"]["drop_rate"] <= 1):
        raise ValueError("dropout rate should be between 0 and 1.")
    if not (0 <= config["model"]["attn_drop_rate"] <= 1):
        raise ValueError("attention dropout rate should be between 0 and 1.")
    if not (0 <= config["model"]["drop_path_rate"] <= 1):
        raise ValueError("drop path rate should be between 0 and 1.")
    if feature_size % 12 != 0:
        raise ValueError("feature_size should be divisible by 12.")

    encoder = SwinTransformer(
            in_chans=config["model"]["in_chans"],
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_sizes,
            depths=config["model"].get("depths", [2, 2, 6, 2]),
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=config["model"]["drop_rate"],
            attn_drop_rate=config["model"]["attn_drop_rate"],
            drop_path_rate=config["model"]["drop_path_rate"],
            norm_layer=nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=spatial_dims,
            downsample="merging",
            use_v2=False,
        )

    if os.path.exists(config["model"]["pretrained_weights"]) and config["model"].get("use_pretrained_weights", False) and not predict_only:
        print(f"Loading pretrained weights from {config['model']['pretrained_weights']}")
        checkpoint = torch.load(config["model"]["pretrained_weights"], weights_only=False)

        pretrained_state_dict = checkpoint['state_dict']
        
        # load in only the swinViT weights
        new_state_dict = {}
        for k, v in pretrained_state_dict.items():
            if k.startswith('module.swinViT.'):
                new_key = k.replace('module.swinViT.', '')
                new_state_dict[new_key] = v

        out = encoder.load_state_dict(new_state_dict, strict=False)
        print(f"Missing Keys: {out.missing_keys}")
        print(f"Unexpected Keys: {out.unexpected_keys}")
        print(f"Length of Missing Keys: {len(out.missing_keys)}")
        print(f"Length of Unexpected Keys: {len(out.unexpected_keys)}")
        print("Pretrained weights loaded successfully.")

    elif predict_only:
        print("Created model for prediction...")

    else:
        print("Training model from scratch")

    # freeze portions of the encoder if needed
    # by default everything is trainable
    set_trainable_swin(
        encoder,
        train_patch_embed=config["model"].get("train_patch_embed", True),
        train_layer1=config["model"].get("train_layer1", True),
        train_layer2=config["model"].get("train_layer2", True),
        train_layer3=config["model"].get("train_layer3", True),
        train_layer4=config["model"].get("train_layer4", True),
        train_all_layernorm=config["model"].get("train_all_layernorm", True),
        layernorm_only=config["model"].get("layernorm_only", False)
    )


    embedding_mode = config["model"].get("embedding_mode", "stage4")
    if embedding_mode == "multiscale":
        feat_dim = feature_size * 31  # 48*(1+2+4+8+16) = 1488
        brain_embedder = TriadSwinViTEmbedder(encoder=encoder, feat_dim=feat_dim)
    else:
        feat_dim = feature_size * 16  # 768
        brain_embedder = SwinViTEmbedder(
            encoder=encoder,
            stage_idx=4,
            pooling=config["model"].get("pooling_method", "gap"),
            feat_dim=feat_dim,
        )

    return _wrap_head(brain_embedder, feat_dim, config)


def create_model_triadswb(config, predict_only):
    feature_size = config["model"].get("feature_size", 48)

    encoder = SwinTransformer(
        in_chans=config["model"]["in_chans"],
        embed_dim=feature_size,
        window_size=ensure_tuple_rep(7, 3),
        patch_size=ensure_tuple_rep(2, 3),
        depths=config["model"].get("depths", [2, 2, 2, 2]),
        num_heads=[3, 6, 12, 24],
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=config["model"]["drop_rate"],
        attn_drop_rate=config["model"]["attn_drop_rate"],
        drop_path_rate=config["model"]["drop_path_rate"],
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        spatial_dims=3,
        downsample="merging",
        use_v2=True,  # Triad SwinB was pretrained with use_v2=True
    )

    if (os.path.exists(config["model"]["pretrained_weights"])
            and config["model"].get("use_pretrained_weights", False)
            and not predict_only):
        print("\n", "-"*80)
        print(f"Loading Triad SwinB weights from {config['model']['pretrained_weights']}")
        raw = torch.load(config["model"]["pretrained_weights"],
                         map_location="cpu", weights_only=False)

        # Checkpoint is a flat state dict with all keys prefixed "backbone.swinViT."
        new_state_dict = {
            k[len("backbone.swinViT."):]: v
            for k, v in raw.items()
            if k.startswith("backbone.swinViT.")
        }

        # inflate patch_embed from 1-channel to in_chans by repeating and rescaling
        pe_key = "patch_embed.proj.weight"
        if pe_key in new_state_dict:
            old_w = new_state_dict[pe_key]  # (embed_dim, 1, 2, 2, 2)
            new_state_dict[pe_key] = old_w.repeat(1, config["model"]["in_chans"], 1, 1, 1) / config["model"]["in_chans"]
            print(f"Channel inflation: {tuple(old_w.shape)} -> {tuple(new_state_dict[pe_key].shape)}")

        out = encoder.load_state_dict(new_state_dict, strict=False)
        print(f"Missing keys ({len(out.missing_keys)}): {out.missing_keys}")
        print(f"Unexpected keys ({len(out.unexpected_keys)}): {out.unexpected_keys}")
        print("-"*80, "\n")

    elif predict_only:
        print("Created Triad SwinB model for prediction...")
    else:
        print("Training Triad SwinB model from scratch")

    set_trainable_swin(
        encoder,
        train_patch_embed=config["model"].get("train_patch_embed", True),
        train_layer1=config["model"].get("train_layer1", True),
        train_layer2=config["model"].get("train_layer2", True),
        train_layer3=config["model"].get("train_layer3", True),
        train_layer4=config["model"].get("train_layer4", True),
        train_all_layernorm=config["model"].get("train_all_layernorm", True),
        layernorm_only=config["model"].get("layernorm_only", False),
    )

    embedding_mode = config["model"].get("embedding_mode", "multiscale")

    if embedding_mode == "stage4":
        # Last stage only — mirrors BrainSeg style (feature_size * 16 = 768 for default feature_size=48)
        feat_dim = feature_size * 16
        embedder = SwinViTEmbedder(
            encoder=encoder,
            stage_idx=4,
            pooling=config["model"].get("pooling_method", "gap"),
            feat_dim=feat_dim,
        )
    else:
        # Default: concat GAP of all 5 skip stages — feature_size * (1+2+4+8+16) = 1488
        feat_dim = feature_size * 31
        embedder = TriadSwinViTEmbedder(encoder=encoder, feat_dim=feat_dim)

    return _wrap_head(embedder, feat_dim, config)


def create_model_brainiac(config, predict_only):
    """BrainIAC ViT-B backbone with mean-pooled modality CLS tokens.

    Each of the M input modalities is processed independently through the
    single-channel ViT; the M CLS tokens are mean-pooled into a single patient
    embedding.  Mirrors the BrainIAC Quad-OS approach.

    Set model.use_projection_head: true in the config to attach the SimCLR
    projection head (768 → 2048-dim), matching the paper's reported feature dim.
    Requires the SimCLR pretrained checkpoint (BrainIAC.ckpt), not the fine-tuned one.

    Checkpoint format: SimCLR Lightning, keys prefixed 'backbone.*'.
    """
    encoder = build_brainiac_vit()

    weights_path = config["model"].get("pretrained_weights", "")
    if os.path.exists(weights_path) and config["model"].get("use_pretrained_weights", False) and not predict_only:
        print("\n", "-" * 80)
        print(f"Loading BrainIAC pretrained weights from {weights_path}")
        load_brainiac_weights(encoder, weights_path)
        print("-" * 80, "\n")
    elif predict_only:
        print("BrainIAC model created for prediction.")
    else:
        print("Training BrainIAC from scratch (no pretrained weights).")

    set_trainable_brainiac(
        encoder,
        frozen_blocks=config["model"].get("frozen_blocks", 0),
        train_patch_embed=config["model"].get("train_patch_embed", True),
        train_norm=config["model"].get("train_all_layernorm", True),
    )

    n_modalities = config["model"].get("in_chans", 4)
    embedding_mode = config["model"].get("embedding_mode", "stage4")

    if embedding_mode == "multiscale":
        if config["model"].get("use_projection_head", False):
            print("WARNING: use_projection_head is ignored when embedding_mode=multiscale.")
        feat_dim = 768 * 4  # CLS tokens from 4 sampled depths × 768, mean-pooled across modalities
        embedder = MultiScaleBrainIACEmbedder(
            encoder=encoder,
            n_modalities=n_modalities,
            feat_dim=feat_dim,
        )
    else:
        use_projection_head = config["model"].get("use_projection_head", False)
        if use_projection_head:
            projection_head = build_brainiac_projection_head(weights_path)
            feat_dim = 2048
        else:
            projection_head = None
            feat_dim = 768
        embedder = BrainIACEmbedder(
            encoder=encoder,
            n_modalities=n_modalities,
            feat_dim=feat_dim,
            projection_head=projection_head,
        )

    return _wrap_head(embedder, feat_dim, config)


# ---------------------------------------------------------------------------
# Embedder-only factory for frozen embedding extraction (extract_embeddings.py)
# Ported from the FM_Extraction sister repo: builds the encoder + embedder with
# pretrained backbone weights but no survival head and no freezing machinery.
# ---------------------------------------------------------------------------

def _check_pretrained(config, allow_untrained):
    """Return True if pretrained weights should be loaded, raise otherwise
    (unless allow_untrained explicitly permits a random-init encoder)."""
    weights_path = config["model"].get("pretrained_weights", "")
    wants_weights = config["model"].get("use_pretrained_weights", True)
    if wants_weights and os.path.exists(weights_path):
        return True
    if allow_untrained:
        print(f"WARNING: pretrained weights not loaded from {weights_path!r} — "
              f"embeddings will come from an untrained encoder (--allow_untrained).")
        return False
    raise FileNotFoundError(
        f"Pretrained weights not found at {weights_path!r} (use_pretrained_weights="
        f"{wants_weights}). Refusing to extract embeddings from an untrained encoder; "
        "fix model.pretrained_weights or pass --allow_untrained."
    )


def create_embedder(config: Dict[str, Any], allow_untrained: bool = False) -> nn.Module:
    """
    Create a bare embedder (no survival head) whose forward pass yields the
    embedding directly. Used for deterministic frozen embedding extraction.
    """
    model_type = config["model"].get("type", "brainmvp").lower()

    if model_type == "brainmvp":
        return create_embedder_brainmvp(config, allow_untrained)
    elif model_type == "brainseg":
        return create_embedder_brainseg(config, allow_untrained)
    elif model_type == "triadswb":
        return create_embedder_triadswb(config, allow_untrained)
    elif model_type == "brainiac":
        return create_embedder_brainiac(config, allow_untrained)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported types: 'brainmvp', 'brainseg', 'triadswb', 'brainiac'"
        )


def create_embedder_brainmvp(config, allow_untrained=False):
    """BrainMVP UniFormer encoder wrapped in an embedder (no head)."""
    # defaulted to match the released BrainMVP weights; overridable in the config
    depths = config["model"].get("depths", [3, 4, 8, 3])
    encoder = UniFormer(
        depth=depths,
        img_size=config["model"]["img_size"],
        in_chans=config["model"]["in_chans"],
        num_classes=1,
    )

    if _check_pretrained(config, allow_untrained):
        weights_path = config["model"]["pretrained_weights"]
        print("\n", "-" * 80)
        print(f"Loading pretrained weights from {weights_path}")
        weights = torch.load(weights_path, map_location="cpu")
        state_dict = {}
        for key in weights["state_dict"].keys():
            new_key = key.replace("module.", "").replace("uniformer.", "").replace("encoder.", "")
            state_dict[new_key] = weights["state_dict"][key]
        out = encoder.load_state_dict(state_dict, strict=False)
        print(f"Missing keys ({len(out.missing_keys)}): {out.missing_keys}")
        print(f"Unexpected keys ({len(out.unexpected_keys)}): {out.unexpected_keys}")
        print("-" * 80, "\n")

    embedding_mode = config["model"].get("embedding_mode", "stage4")
    if embedding_mode == "multiscale":
        feat_dim = 1024  # 64+128+320+512 across UniFormer stages 1-4
        return MultiScaleUniformerEmbedder(encoder=encoder, feat_dim=feat_dim)

    return UniformerEmbedder(
        encoder=encoder,
        stage_idx=4,
        pooling=config["model"].get("pooling_method", "gap"),
        feat_dim=512,
    )


def create_embedder_brainseg(config, allow_untrained=False):
    """BrainSegFounder Swin encoder wrapped in an embedder (no head)."""
    # hardcoded values to match the released weights
    spatial_dims = 3
    feature_size = 48
    encoder = SwinTransformer(
        in_chans=config["model"]["in_chans"],
        embed_dim=feature_size,
        window_size=ensure_tuple_rep(7, spatial_dims),
        patch_size=ensure_tuple_rep(2, spatial_dims),
        depths=config["model"].get("depths", [2, 2, 6, 2]),
        num_heads=[3, 6, 12, 24],
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        spatial_dims=spatial_dims,
        downsample="merging",
        use_v2=False,
    )

    if _check_pretrained(config, allow_untrained):
        weights_path = config["model"]["pretrained_weights"]
        print(f"Loading pretrained weights from {weights_path}")
        checkpoint = torch.load(weights_path, weights_only=False)
        # load in only the swinViT weights
        new_state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            if k.startswith("module.swinViT."):
                new_state_dict[k.replace("module.swinViT.", "")] = v
        out = encoder.load_state_dict(new_state_dict, strict=False)
        print(f"Missing keys ({len(out.missing_keys)}): {out.missing_keys}")
        print(f"Unexpected keys ({len(out.unexpected_keys)}): {out.unexpected_keys}")

    embedding_mode = config["model"].get("embedding_mode", "stage4")
    if embedding_mode == "multiscale":
        feat_dim = feature_size * 31  # 48*(1+2+4+8+16) = 1488
        return TriadSwinViTEmbedder(encoder=encoder, feat_dim=feat_dim)
    return SwinViTEmbedder(
        encoder=encoder,
        stage_idx=4,
        pooling=config["model"].get("pooling_method", "gap"),
        feat_dim=feature_size * 16,  # 768
    )


def create_embedder_triadswb(config, allow_untrained=False):
    """Triad SwinB encoder wrapped in an embedder (no head)."""
    feature_size = config["model"].get("feature_size", 48)
    encoder = SwinTransformer(
        in_chans=config["model"]["in_chans"],
        embed_dim=feature_size,
        window_size=ensure_tuple_rep(7, 3),
        patch_size=ensure_tuple_rep(2, 3),
        depths=config["model"].get("depths", [2, 2, 2, 2]),
        num_heads=[3, 6, 12, 24],
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        spatial_dims=3,
        downsample="merging",
        use_v2=True,  # Triad SwinB was pretrained with use_v2=True
    )

    if _check_pretrained(config, allow_untrained):
        weights_path = config["model"]["pretrained_weights"]
        print("\n", "-" * 80)
        print(f"Loading Triad SwinB weights from {weights_path}")
        raw = torch.load(weights_path, map_location="cpu", weights_only=False)
        # checkpoint is a flat state dict with all keys prefixed "backbone.swinViT."
        new_state_dict = {
            k[len("backbone.swinViT."):]: v
            for k, v in raw.items()
            if k.startswith("backbone.swinViT.")
        }
        out = encoder.load_state_dict(new_state_dict, strict=False)
        print(f"Missing keys ({len(out.missing_keys)}): {out.missing_keys}")
        print(f"Unexpected keys ({len(out.unexpected_keys)}): {out.unexpected_keys}")
        print("-" * 80, "\n")

    embedding_mode = config["model"].get("embedding_mode", "multiscale")
    if embedding_mode == "stage4":
        # last stage only — feature_size * 16 = 768 for default feature_size=48
        return SwinViTEmbedder(
            encoder=encoder,
            stage_idx=4,
            pooling=config["model"].get("pooling_method", "gap"),
            feat_dim=feature_size * 16,
        )
    # default: concat GAP of all 5 skip stages — feature_size * (1+2+4+8+16) = 1488
    return TriadSwinViTEmbedder(encoder=encoder, feat_dim=feature_size * 31)


def create_embedder_brainiac(config, allow_untrained=False):
    """BrainIAC ViT-B backbone with mean-pooled modality CLS tokens (no head)."""
    encoder = build_brainiac_vit()

    if _check_pretrained(config, allow_untrained):
        weights_path = config["model"]["pretrained_weights"]
        print("\n", "-" * 80)
        print(f"Loading BrainIAC pretrained weights from {weights_path}")
        load_brainiac_weights(encoder, weights_path)
        print("-" * 80, "\n")

    n_modalities = config["model"].get("in_chans", 4)
    embedding_mode = config["model"].get("embedding_mode", "stage4")

    if embedding_mode == "multiscale":
        if config["model"].get("use_projection_head", False):
            print("WARNING: use_projection_head is ignored when embedding_mode=multiscale.")
        feat_dim = 768 * 4  # CLS tokens from 4 sampled depths x 768, mean-pooled across modalities
        return MultiScaleBrainIACEmbedder(
            encoder=encoder,
            n_modalities=n_modalities,
            feat_dim=feat_dim,
        )

    if config["model"].get("use_projection_head", False):
        projection_head = build_brainiac_projection_head(config["model"].get("pretrained_weights", ""))
        feat_dim = 2048
    else:
        projection_head = None
        feat_dim = 768
    return BrainIACEmbedder(
        encoder=encoder,
        n_modalities=n_modalities,
        feat_dim=feat_dim,
        projection_head=projection_head,
    )
