import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention (W-MSA).

    Implements the core attention mechanism used in Swin Transformer.
    Computes attention within local windows for efficiency.

    Args:
        dim: Input dimension (number of channels)
        window_size: Window size (M)
        num_heads: Number of attention heads
    """
    def __init__(self, dim, window_size=8, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Args:
            x: (B*num_windows, M*M, C) - flattened windows
        Returns:
            (B*num_windows, M*M, C)
        """
        B_, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention: Q @ K^T / sqrt(d)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B_, num_heads, N, N)
        attn = F.softmax(attn, dim=-1)

        # Attention @ V
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        return x
    