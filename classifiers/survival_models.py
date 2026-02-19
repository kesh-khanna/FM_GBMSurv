from abc import ABC, abstractmethod
from typing import Dict, List
import torch.nn as nn
from embedders.base_embedders import BaseEmbedder

class BasePredictionModel(nn.Module, ABC):
    """
    BasePredictionModel to be used to make sure that prediction models
      have a function for getting param groups
      Could also be used down the line for a function like getting embeddings
      or some kind of feature importance analysis ect...

    """
    def __init__(self):
        
        super().__init__()
    
    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        """
        Organizes parameters organized into groups for differential learning rates.
    
        includes a dict with param groups, must include at least:
        - "backbone": encoder/pretrained parameters
        - "head": task-specific parameters
        
        Can also include "pooling": pooling layer parameters (if they are present)
        """
        pass


class DeepSurvNet(BasePredictionModel):
    def __init__(self, embedder: BaseEmbedder, embedding_dim=512, hidden_dims=[256], 
                 return_embeddings=False):
        super().__init__()
        self.return_embeddings = return_embeddings
        self.embedder = embedder
        
        # MLP on top of pooled embeddings
        layers = []
        prev_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        embedding = self.embedder(x)  # [B, embedding_dim]

        out = self.net(embedding) # [embedding dim -> ... -> 1 (log_hz if using nll coxph loss)]
        
        if self.return_embeddings:
            return out, embedding
        else:
            return out
    
    def get_param_groups(self):
        embedder_groups = self.embedder.get_param_groups()
    
        embedder_groups["head"] = list(self.net.parameters())

        return embedder_groups
    
