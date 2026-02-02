import torch.nn as nn
import torch.nn.functional as F

class GlobalTransformerBlock(nn.Module):
    """
    Standard Transformer block with global attention.

    Used at the bottleneck where spatial dimensions are small,
    so global attention is computationally feasible.

    From paper Eq. (4):
    X̂ˡ = MSA(LN(Xˡ⁻¹)) + Xˡ⁻¹
    Xˡ = MLP(LN(X̂ˡ)) + X̂ˡ

    Args:
        dim: Channel dimension
        num_heads: Number of attention heads
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # MLP
        mlp_hidden_dim = int(dim * 4)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        B, C, H, W = x.shape
        shortcut = x

        # Reshape to (B, N, C) where N = H*W
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        N = x.shape[1]

        # Multi-head self-attention
        x = self.norm1(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        # Reshape back and residual
        x = x.transpose(1, 2).view(B, C, H, W)
        x = shortcut + x

        # MLP block
        shortcut = x
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm2(x)
        x = self.mlp(x)
        x = x.transpose(1, 2).view(B, C, H, W)

        # Residual
        x = shortcut + x

        return x
