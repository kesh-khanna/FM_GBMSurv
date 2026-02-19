

"""
Model-specific encoder implementations
"""
import torch
from typing import Optional
from embedders.base_embedders import BaseEmbedder
from embedders.pooling import FeatureMapPool

class UniformerEmbedder(BaseEmbedder):
    """
    Takes the output of a uniformer and pools it into a fixed length vector for downstream prediction
    Input:  x [B, 4, H, W, D]
    Output:
      z_patient [B, 512]
    """
    def __init__(
        self,
        encoder,
        stage_idx=4,            # which stage of teh model to pull from
        pooling="gap",       # "gap" | "gem" | "max"
        feat_dim=512,
    ):
        pooling = FeatureMapPool(kind=pooling)
        super().__init__(embedding_dim=feat_dim, encoder=encoder, pooling=pooling)
        self.stage_idx = stage_idx
    

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected [B,M,H,W,D], got {tuple(x.shape)}")

        B, M, H, W, D = x.shape # expect [B, 4, H, W, D]
        
        x_4 = self.encoder(x)[self.stage_idx]  # [B, M, H, W, D] -> [B, C, D', H', W']

        # token pool 
        feats = self.pooling(x_4)  # -> [B, C]

        return feats
    
    def get_intermediate_features(self, x: torch.Tensor) -> dict:
        outputs = self.encoder(x)
        return {
            f'stage_{i}': out for i, out in enumerate(outputs)
        }
    
    def get_param_groups(self):
        """
        backbone: Uniformer
        pooling: any features from the feature map pooling (like gem or future attn based methods)
        """
        return super().get_param_groups()
    

class SwinViTEmbedder(BaseEmbedder):
    """
    Meant to be connected to the SwinViT encoder
    Takes the output of swin and pools it into a fixed length vector for downstream prediction
    Input:  x [B, 4, H, W, D]
    Output:
      z_patient [B, 768]
    """
    def __init__(
        self,
        encoder,
        stage_idx=4, 
        pooling="gap",
        feat_dim: int =768,
        normalize: bool=True
    ):
        pooling = FeatureMapPool(kind=pooling)
        super().__init__(embedding_dim=feat_dim, encoder=encoder, pooling=pooling)
        self.stage_idx = stage_idx
        self.normalize = normalize
    
    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected [B,M,H,W,D], got {tuple(x.shape)}")

        B, M, H, W, D = x.shape # expect [B, 4, H, W, D]
        
        x_4 = self.encoder(x.contiguous(), normalize=self.normalize)[self.stage_idx]  # [B, M, H, W, D] -> [B, C, D', H', W']

        # pool featuure map 
        feats = self.pooling(x_4)  # -> [B, C]

        return feats
    
    def get_intermediate_features(self, x: torch.Tensor) -> dict:
        features = self.encoder(x, normalize=self.normalize)
        return {
            f'stage_{i}': feat for i, feat in enumerate(features)
        }

    def get_param_groups(self):
        """
        backbone: SwinViT
        pooling: any features from the feature map pooling (like gem or future attn based methods)
        """
        return super().get_param_groups()
    
    
