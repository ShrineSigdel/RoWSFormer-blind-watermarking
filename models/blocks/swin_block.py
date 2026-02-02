import torch
import torch.nn as nn

from models.blocks.attention import WindowAttention
from models.layers.utils import window_partition, window_reverse


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block with window-based attention.

    From paper Eq. (3):
    X̂ˡ = W-MSA(LN(Xˡ⁻¹)) + Xˡ⁻¹
    Xˡ = MLP(LN(X̂ˡ)) + X̂ˡ

    Args:
        dim: Channel dimension
        num_heads: Number of attention heads
        window_size: Window size for attention
        shift_size: Shift size for shifted window attention (0 for W-MSA)
    """
    def __init__(self, dim, num_heads=4, window_size=8, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        # Normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Window attention
        self.attn = WindowAttention(dim, window_size, num_heads)

        # MLP: 2-layer with GELU
        mlp_hidden_dim = int(dim * 4)  # Standard expansion ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x, H, W):
        """
        Args:
            x: (B, C, H, W)
            H, W: Spatial dimensions
        Returns:
            (B, C, H, W)
        """
        B, C, H, W = x.shape
        shortcut = x

        # Reshape to (B, H*W, C) for attention
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Cyclic shift (for shifted   window attention)
        if self.shift_size > 0:
            x = x.view(B, H, W, C)
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            x = x.view(B, H * W, C)

        # Partition into windows
        x = x.view(B, H, W, C)
        x_windows = window_partition(
            x.permute(0, 3, 1, 2).contiguous(),
            self.window_size
        )  # (B*num_windows, C, M, M)

        # Flatten windows for attention
        x_windows = x_windows.flatten(2).transpose(1, 2)  # (B*num_windows, M*M, C)

        # W-MSA
        x_windows = self.norm1(x_windows)
        attn_windows = self.attn(x_windows)  # (B*num_windows, M*M, C)

        # Reverse window partition
        attn_windows = attn_windows.transpose(1, 2).view(-1, C, self.window_size, self.window_size)
        x = window_reverse(attn_windows, self.window_size, H, W)  # (B, C, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # First residual connection
        x = shortcut + x

        # MLP block
        shortcut = x
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.norm2(x)
        x = self.mlp(x)
        x = x.transpose(1, 2).view(B, C, H, W)

        # Second residual connection
        x = shortcut + x

        return x
