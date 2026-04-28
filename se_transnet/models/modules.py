"""
Reusable model building blocks for SE-TransNet.

VarPool2D, PositionalEncoding, MultiHeadedAttention, FeedForward, TransformerEncoder.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class VarPool2D(nn.Module):
    """Log-variance pooling over temporal windows using torch.unfold."""

    def __init__(self, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        windows = x.unfold(-1, self.kernel_size, self.stride)
        var_val = windows.float().var(dim=-1)
        log_var = torch.log(
            torch.clamp(var_val, min=1e-6, max=1e6)
        ).to(x.dtype)
        return log_var


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 200) -> None:
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class MultiHeadedAttention(nn.Module):
    """Multi-head self-attention utilizing highly-optimized PyTorch 2.0 FlashAttention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(self, query, key, value):
        q = rearrange(self.w_q(query), 'b n (h d) -> b h n d', h=self.n_heads)
        k = rearrange(self.w_k(key), 'b n (h d) -> b h n d', h=self.n_heads)
        v = rearrange(self.w_v(value), 'b n (h d) -> b h n d', h=self.n_heads)
        
        # PyTorch 2.0 FlashAttention / Memory-Efficient Attention core
        out = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout_p if self.training else 0.0
        )
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.w_o(out)


class FeedForward(nn.Module):
    """Position-wise FFN with GELU activation."""

    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.act = nn.GELU()
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.w_2(self.dropout(self.act(self.w_1(x)))))


class TransformerEncoder(nn.Module):
    """Pre-norm Transformer encoder block."""

    def __init__(
        self, embed_dim: int, num_heads: int,
        fc_ratio: int = 4, attn_drop: float = 0.1, fc_drop: float = 0.1,
    ) -> None:
        super().__init__()
        self.mha = MultiHeadedAttention(embed_dim, num_heads, attn_drop)
        self.ff = FeedForward(embed_dim, embed_dim * fc_ratio, fc_drop)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = self.layernorm1(x)
        x = x + self.mha(residual, residual, residual)
        residual = self.layernorm2(x)
        x = x + self.ff(residual)
        return x