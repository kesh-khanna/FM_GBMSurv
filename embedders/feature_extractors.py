

"""
Model-specific encoder implementations
"""
import torch
import torch.nn as nn
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
    
    def extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return stage-4 spatial tokens before pooling: [B, D'*H'*W', C]."""
        x_stage = self.encoder(x)[self.stage_idx]           # [B, C, D', H', W']
        B, C, D, H, W = x_stage.shape
        return x_stage.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)

    def get_param_groups(self):
        """
        backbone: Uniformer
        pooling: any features from the feature map pooling (like gem or future attn based methods)
        """
        return super().get_param_groups()


class MultiScaleUniformerEmbedder(BaseEmbedder):
    """GAP-pools UniFormer stages 1-4 and concatenates.
    encoder(x) returns (x_raw, x1, x2, x3, x4); index 0 is the permuted raw
    input — skip it. Default embed_dims=[64,128,320,512] → 1024-dim output.
    """
    def __init__(self, encoder, feat_dim: int = 1024):
        pooling = FeatureMapPool(kind="gap")
        super().__init__(embedding_dim=feat_dim, encoder=encoder, pooling=pooling)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected [B,M,H,W,D], got {tuple(x.shape)}")
        stages = self.encoder(x)                                    # (x_raw, x1, x2, x3, x4)
        pooled = [s.mean(dim=(2, 3, 4)) for s in stages[1:]]       # skip raw input at index 0
        return torch.cat(pooled, dim=1)                             # [B, 1024]

    def extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return final-stage spatial tokens before pooling: [B, D'*H'*W', C]."""
        x_stage = self.encoder(x)[4]                        # [B, C, D', H', W'] — stage 4
        B, C, D, H, W = x_stage.shape
        return x_stage.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)

    def get_intermediate_features(self, x):
        stages = self.encoder(x)
        return {f'stage_{i}': s for i, s in enumerate(stages[1:], start=1)}

    def get_param_groups(self):
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

    def extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return stage-{stage_idx} spatial tokens before pooling: [B, D'*H'*W', C]."""
        x_stage = self.encoder(x.contiguous(), normalize=self.normalize)[self.stage_idx]
        B, C, D, H, W = x_stage.shape
        return x_stage.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)

    def get_param_groups(self):
        """
        backbone: SwinViT
        pooling: any features from the feature map pooling (like gem or future attn based methods)
        """
        return super().get_param_groups()


class TriadSwinViTEmbedder(BaseEmbedder):
    """
    Multi-scale embedder for the Triad SwinB backbone.
    GAP-pools all 5 skip stages from SwinTransformer.forward() and concatenates.
    For embed_dim=48: (48+96+192+384+768) = 1488-dim output.
    """
    def __init__(self, encoder, feat_dim: int = 1488, normalize: bool = True):
        pooling = FeatureMapPool(kind="gap")  # no learned params; GAP is applied inline
        super().__init__(embedding_dim=feat_dim, encoder=encoder, pooling=pooling)
        self.normalize = normalize

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Expected [B,M,H,W,D], got {tuple(x.shape)}")
        stages = self.encoder(x.contiguous(), normalize=self.normalize)
        pooled = [s.mean(dim=(2, 3, 4)) for s in stages]
        return torch.cat(pooled, dim=1)

    def extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return final-stage (index 4) spatial tokens before pooling: [B, D'*H'*W', C]."""
        x_stage = self.encoder(x.contiguous(), normalize=self.normalize)[4]
        B, C, D, H, W = x_stage.shape
        return x_stage.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)

    def get_intermediate_features(self, x):
        features = self.encoder(x.contiguous(), normalize=self.normalize)
        return {f'stage_{i}': feat for i, feat in enumerate(features)}

    def get_param_groups(self):
        return super().get_param_groups()


class BrainIACEmbedder(BaseEmbedder):
    """Mean-pools M independently-encoded modality CLS tokens → [B, feat_dim].

    The BrainIAC ViT is strictly single-channel (in_channels=1).  Each of the
    M modalities is fed through the shared ViT separately; the M CLS tokens are
    stacked and averaged.  This mirrors the BrainIAC Quad-OS downstream approach.

    If projection_head is provided, each CLS token is passed through it before
    averaging, yielding 2048-dim output instead of 768-dim (matching the paper).

    Input:  x [B, M, H, W, D]   (M = in_chans from config, typically 4)
    Output: z [B, feat_dim]      (768 without projection head, 2048 with)
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_modalities: int = 4,
        feat_dim: int = 768,
        projection_head: Optional[nn.Module] = None,
    ):
        # No feature-map pooling — CLS token is the embedding.
        # Passing pooling=None; get_param_groups is overridden below.
        super().__init__(embedding_dim=feat_dim, encoder=encoder, pooling=None)
        self.n_modalities = n_modalities
        self.projection_head = projection_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B, M, H, W, D], got {tuple(x.shape)}")
        cls_tokens = []
        for i in range(x.shape[1]):
            hidden, _ = self.encoder(x[:, i : i + 1])  # [B, N+1, 768]
            cls = hidden[:, 0]                          # [B, 768] CLS token
            if self.projection_head is not None:
                cls = self.projection_head(cls)         # [B, 2048]
            cls_tokens.append(cls)
        return torch.stack(cls_tokens, dim=1).mean(dim=1)  # [B, feat_dim]

    def extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return all patch tokens (excl. CLS) from final ViT layer, concatenated across modalities.
        Output: [B, M * N_patches, hidden_size].
        """
        all_tokens = []
        for i in range(x.shape[1]):
            hidden, _ = self.encoder(x[:, i : i + 1])  # [B, N+1, hidden_size]
            all_tokens.append(hidden[:, 1:])            # [B, N_patches, hidden_size]
        return torch.cat(all_tokens, dim=1)             # [B, M*N_patches, hidden_size]

    def get_intermediate_features(self, x: torch.Tensor) -> dict:
        """Returns per-modality embeddings (after projection head if set) before mean pooling."""
        result = {}
        for i in range(x.shape[1]):
            hidden, _ = self.encoder(x[:, i : i + 1])
            cls = hidden[:, 0]
            if self.projection_head is not None:
                cls = self.projection_head(cls)
            result[f"modality_{i}_cls"] = cls
        return result

    def get_param_groups(self) -> dict:
        params = list(self.encoder.parameters())
        if self.projection_head is not None:
            params += list(self.projection_head.parameters())
        return {"backbone": params, "pooling": []}


class MultiScaleBrainIACEmbedder(BaseEmbedder):
    """Extracts CLS tokens at 4 intermediate ViT depths, concatenates, mean-pools modalities.

    MONAI ViT returns (final_hidden, hidden_states_out) where hidden_states_out is a
    list of num_layers tensors [B, N+1, hidden_size]. The current BrainIACEmbedder
    discards hidden_states_out; this class uses it.

    Default sample_blocks=(2, 5, 8, 11) → every 3rd layer of ViT-B/12 (0-indexed).
    Output per modality: len(sample_blocks) × 768 = 3072-dim.
    Mean-pooled across M modalities → [B, 3072].
    """
    def __init__(
        self,
        encoder: nn.Module,
        n_modalities: int = 4,
        feat_dim: int = 3072,
        sample_blocks: tuple = (2, 5, 8, 11),
    ):
        super().__init__(embedding_dim=feat_dim, encoder=encoder, pooling=None)
        self.n_modalities = n_modalities
        self.sample_blocks = list(sample_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B, M, H, W, D], got {tuple(x.shape)}")
        cls_tokens = []
        for i in range(x.shape[1]):
            _, hidden_states = self.encoder(x[:, i : i + 1])           # list of num_layers × [B, N+1, 768]
            per_depth = [hidden_states[k][:, 0] for k in self.sample_blocks]   # each [B, 768]
            cls_tokens.append(torch.cat(per_depth, dim=1))              # [B, 3072]
        return torch.stack(cls_tokens, dim=1).mean(dim=1)               # [B, 3072]

    def extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Delegate to final-layer patch tokens (same as BrainIACEmbedder.extract_tokens)."""
        all_tokens = []
        for i in range(x.shape[1]):
            hidden, _ = self.encoder(x[:, i : i + 1])
            all_tokens.append(hidden[:, 1:])
        return torch.cat(all_tokens, dim=1)

    def get_intermediate_features(self, x: torch.Tensor) -> dict:
        result = {}
        for i in range(x.shape[1]):
            _, hidden_states = self.encoder(x[:, i : i + 1])
            for k in self.sample_blocks:
                result[f"modality_{i}_block_{k}_cls"] = hidden_states[k][:, 0]
        return result

    def get_param_groups(self) -> dict:
        return {"backbone": list(self.encoder.parameters()), "pooling": []}

