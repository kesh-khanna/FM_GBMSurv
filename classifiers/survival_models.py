import torch.nn as nn
from embedders import BaseEmbedder
class DeepSurvNet(nn.Module):
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
        backbone_params = list(self.embedder.encoder.parameters()) # embedder should also have a get params
        head_params = []

        for p in self.parameters():
            if p not in backbone_params:
                head_params.append(p)

        backbone_params = [p for p in backbone_params if p.requires_grad]
        head_params = [p for p in head_params if p.requires_grad]

        # Safety: ensure no duplicates
        assert len(set(backbone_params + head_params)) == \
            len(backbone_params) + len(head_params), \
            "Duplicate params"

        return {
            "backbone": backbone_params,
            "head": head_params,
        }