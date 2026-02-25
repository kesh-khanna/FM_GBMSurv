"""
Author: Rakesh Khanna
"""

from backbones.uniformer import UniFormer, set_trainable_uniformer
from backbones.swin_encoder import SwinTransformer, set_trainable_swin
from embedders.feature_extractors import UniformerEmbedder, SwinViTEmbedder
from classifiers.survival_models import DeepSurvNet

from typing import Dict, Any
import torch.nn as nn
import os
import torch

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
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Supported types: 'brainmvp', 'brainseg'"
            )

def create_model_brainmvp(config, predict_only):
    """
    create the encoder and wrap the encoder in the embedding model. could change to pass configs individually
    """
    # currently defaulted to match the weights of the BrainMVP, can be overwritten in the config if needed
    depths = config["model"].get("depths", [3, 4, 8, 3])
    encoder = UniFormer(depth=depths, img_size=config["model"]["img_size"], in_chans=config["model"]["in_chans"], num_classes=1)

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

    brain_embedder = UniformerEmbedder(
        encoder=encoder,
        stage_idx=4,
        pooling=config["model"].get("pooling_method", "gap"),
        feat_dim=512,
    )

    model = DeepSurvNet(
        embedder=brain_embedder,
        embedding_dim=512,
        hidden_dims=config["model"].get("hidden_dims", [256]),
        return_embeddings=config["model"].get("return_embeddings", False)
    )
    return model

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
    elif predict_only: 
        print("Created model for prediction...")
        
    else:
        print("Training model from scratch")
    
    
    brain_embedder = SwinViTEmbedder(
        encoder=encoder,
        stage_idx=4,
        pooling=config["model"].get("pooling_method", "gap"),
        feat_dim=feature_size * 16,
    )

    model = DeepSurvNet(
        embedder=brain_embedder,
        embedding_dim=feature_size * 16,
        hidden_dims=config["model"].get("hidden_dims", [int(feature_size * 4)]), # squeeze of params down to 192 if 48 fs
        return_embeddings=config["model"].get("return_embeddings", False)
    )

    return model
