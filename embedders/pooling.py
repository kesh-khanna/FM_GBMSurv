import torch
import torch.nn as nn
from abc import ABC, abstractmethod


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
