from __future__ import annotations

"""
SE-TransNet: SEED-IV Emotion TransNet.

Full shape trace:
  [B,62,800] -> [B,1,62,800] -> [B,80,62,800] -> [B,80,1,800] ->
  avg:[B,80,77]  var:[B,80,77] -> [B,77,80]x2 -> (+PE) -> SA(N=6) ->
  [B,77,2,80] -> ConvEnc -> [B,64,1,80] -> [B,5120] ->
  [B,256] -> [B,64] -> [B,4]
"""

import torch
import torch.nn as nn
from einops import rearrange

from se_transnet.models.modules import (
    PositionalEncoding, TransformerEncoder, VarPool2D,
)


class SETransNet(nn.Module):
    """SE-TransNet: Emotion-adapted EEG-TransNet for SEED-IV.

    Args:
        num_classes: 4 (Neutral, Sad, Fear, Happy).
        num_samples: 800 (4s x 200Hz).
        num_channels: 62 (SEED-IV ESI NeuroScan).
        embed_dim: 80 (5 branches x F1=16).
        temporal_kernels: [25,51,101,201,401] for emotion dynamics.
        pool_size: 40 (200ms @ 200Hz).
        pool_stride: 10 (50ms hop -> T_seq=77).
        num_heads: 8 (80/8=10 per head).
        fc_ratio: 4 (FFN hidden = 80x4=320).
        depth: 6 transformer layers.
        attn_drop: 0.1, fc_drop: 0.5, spatial_drop: 0.25.
    """

    def __init__(
        self,
        num_classes: int = 4,
        num_samples: int = 800,
        num_channels: int = 62,
        embed_dim: int = 80,
        temporal_kernels: list = None,
        pool_size: int = 40,
        pool_stride: int = 10,
        num_heads: int = 8,
        fc_ratio: int = 4,
        depth: int = 6,
        attn_drop: float = 0.1,
        fc_drop: float = 0.5,
        spatial_drop: float = 0.25,
    ) -> None:
        super().__init__()

        if temporal_kernels is None:
            temporal_kernels = [25, 51, 101, 201, 401]

        n_branches = len(temporal_kernels)
        assert embed_dim % n_branches == 0
        F1 = embed_dim // n_branches  # 16

        # -- Multi-scale Temporal Filter Bank --
        self.temporal_convs = nn.ModuleList([
            nn.Conv2d(1, F1, kernel_size=(1, k),
                      padding=(0, k // 2), bias=False)
            for k in temporal_kernels
        ])
        self.bn_temporal = nn.BatchNorm2d(embed_dim)

        # -- Depthwise Separable Spatial Conv --
        self.spatial_dw = nn.Conv2d(
            embed_dim, embed_dim,
            kernel_size=(num_channels, 1),
            groups=embed_dim, bias=False,
        )
        self.spatial_pw = nn.Conv2d(
            embed_dim, embed_dim,
            kernel_size=(1, 1), bias=False,
        )
        self.bn_spatial = nn.BatchNorm2d(embed_dim)
        self.elu = nn.ELU()
        self.spatial_dropout = nn.Dropout(p=spatial_drop)

        # -- Dual Temporal Pooling --
        T_seq = (num_samples - pool_size) // pool_stride + 1
        self.avg_pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_stride)
        self.var_pool = VarPool2D(kernel_size=pool_size, stride=pool_stride)
        self.pool_dropout = nn.Dropout(p=0.1)

        # -- Positional Encoding --
        self.pos_enc = PositionalEncoding(d_model=embed_dim, max_len=T_seq + 10)

        # -- Shared Self-Attention Stack --
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(
                embed_dim=embed_dim, num_heads=num_heads,
                fc_ratio=fc_ratio, attn_drop=attn_drop,
                fc_drop=attn_drop,
            )
            for _ in range(depth)
        ])

        # -- 2-Layer Convolutional Encoder --
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(T_seq, 64, kernel_size=(2, 1), bias=False),
            nn.BatchNorm2d(64), nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(64), nn.ELU(),
            nn.Dropout(p=0.25),
        )

        # -- 3-Layer Classification Head --
        fc_in = 64 * embed_dim  # 5120
        self.classifier = nn.Sequential(
            nn.Linear(fc_in, 256),
            nn.BatchNorm1d(256), nn.ELU(),
            nn.Dropout(p=fc_drop),
            nn.Linear(256, 64),
            nn.ELU(), nn.Dropout(p=0.3),
            nn.Linear(64, num_classes),
        )
        self._fc_in = fc_in

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features at FC1 output (B, 256) for domain adaptation."""
        feats = self._forward_backbone(x)
        for layer in list(self.classifier.children())[:4]:
            feats = layer(feats)
        return feats

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone: input -> flatten before classifier."""
        x = x.unsqueeze(dim=1)                           # (B,1,62,800)
        branches = [conv(x) for conv in self.temporal_convs]
        x = self.bn_temporal(torch.cat(branches, dim=1))  # (B,80,62,800)

        x = self.spatial_dw(x)                             # (B,80,1,800)
        x = self.spatial_pw(x)
        x = self.spatial_dropout(self.elu(self.bn_spatial(x)))
        x = x.squeeze(dim=2)                               # (B,80,800)

        x_avg = self.pool_dropout(self.avg_pool(x))         # (B,80,77)
        x_var = self.pool_dropout(self.var_pool(x))          # (B,80,77)

        x_avg = rearrange(x_avg, 'b d n -> b n d')          # (B,77,80)
        x_var = rearrange(x_var, 'b d n -> b n d')

        x_avg = self.pos_enc(x_avg)
        x_var = self.pos_enc(x_var)

        for enc in self.transformer_encoders:
            x_avg = enc(x_avg)
            x_var = enc(x_var)

        x_avg = x_avg.unsqueeze(dim=2)                      # (B,77,1,80)
        x_var = x_var.unsqueeze(dim=2)
        x = torch.cat([x_avg, x_var], dim=2)                # (B,77,2,80)
        x = self.conv_encoder(x)                             # (B,64,1,80)
        x = x.reshape(x.size(0), -1)                        # (B,5120)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward: (B,62,800) -> (B,4) logits."""
        x = self._forward_backbone(x)
        return self.classifier(x)
