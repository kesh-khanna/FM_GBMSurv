

"""
Model-specific encoder implementations
"""
import torch
import torch.nn as nn
from typing import Optional
from embedders.base_embedders import BaseEmbedder
from abc import ABC, abstractmethod


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
    
    
class BasePooling(nn.Module, ABC):
    """
    Abstract class for the pooling operations needed to generate embeddings
    Takes outputs from a given model and pools them to a fixed length vector
    """
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input: Tensor, shape dependent on backbone output
        returns: Tensor of shape [B, C]
        """
        pass

class FeatureMapPool(BasePooling):
    """
    Pools the spatial feature maps that are provided by SwinViT and the Uniformer Architectures
    [B,C,d',h',w'] -> [B,C]
    """

    def __init__(self, kind="gap", gem_p=3.0, eps=1e-6):
        super().__init__()
        self.kind = kind
        self.eps = eps

        if kind == "gem":
            self.p = nn.Parameter(torch.ones(1) * gem_p)

    def forward(self, x):
        if self.kind == "gap":
            return x.mean(dim=(2, 3, 4))
        
        elif self.kind == "max":
            return x.amax(dim=(2, 3, 4))
        
        elif self.kind == "gem":
            x = x.clamp(min=self.eps).pow(self.p)
            return x.mean(dim=(2, 3, 4)).pow(1.0 / self.p)
        
        else:
            raise ValueError(f"Unknown feature pool method: {self.kind}")
