
"""
Abstract base classes for flexible multi-model architecture
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from embedders import BasePooling


class BaseEmbedder(nn.Module, ABC):
    """
    Abstract base class for all embedding heads and associated backbones.
    Ensures consistent interface across BrainMVP, BrainSegFounder, and BrainIAC.
    """
    
    def __init__(self, embedding_dim: int, encoder: nn.Module, pooling: BasePooling):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder = encoder
        self.pooling = pooling
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, 4, H, W, D] (4 modalities)
            
        Returns:
            embeddings: Tensor of shape [B, embedding_dim]
        """
        pass
    
    @abstractmethod
    def get_intermediate_features(self, x: torch.Tensor) -> dict:
        """
        intermediate features for visualization/analysis.
        
        Args:
            x: Input tensor [B, 4, H, W, D]
            
        Returns:
            dict with intermediate features
        """
        pass

    def get_param_groups(self) -> dict:
        """
        get the parameters that are part of the backbone and separates and pooling parameters into "pooling"

        Returns:
            dict with param groups
            "backbone"
            "pooling"
        """
        return {
            "backbone": list(self.encoder.parameters()),
            "pooling": list(self.pooling.parameters())
        }
