import torch.nn as nn

class DeepSurvNet(nn.Module):
    def __init__(self, embedder: nn.Module, embedding_dim=512, hidden_dims=[256], 
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