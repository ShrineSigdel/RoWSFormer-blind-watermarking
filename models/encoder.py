import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """
    Converts image to patch embeddings.
    
    Args:
        in_channels: Input channels (3 for RGB)
        embed_dim: Embedding dimension (C)
        patch_size: Size of each patch (P)
    """
    def __init__(self, in_channels=3, embed_dim=64, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Simple conv layer to extract features
        # From paper: "we first apply a 3×3 convolutional layer"
        self.conv = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=3, padding=1, stride=1)
    
    # convolution ko feed forward layer jastai use bhacha
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            features: (B, C, H, W)
        """
        # Extract features
        features = self.conv(x)  # (B, C, H, W)
        return features


class WatermarkEncoder(nn.Module):
    """
    Encodes watermark bits and expands them spatially.

    Process:
    1. Bits (L,) → Linear → (L1,)
    2. Reshape → (L2, L2)
    3. Upsample → (H, W) to match image features
    4. Conv → (C1,) channels

    Args:
        watermark_length: Number of bits (64)
        embed_dim: Output channels (C1)
        image_size: Target spatial size (128)
    """
    def __init__(self, watermark_length=64, embed_dim=32, image_size=128):
        super().__init__()
        self.watermark_length = watermark_length
        self.embed_dim = embed_dim
        self.image_size = image_size

        # From paper: "Men passes through a linear layer that produces
        # an output vector of length L1. This vector is then reshaped
        # into a matrix of size L2×L2."

        # We'll use a simple approach: bits → small spatial map → upsample
        # L1 = 16*16 = 256 (for example)
        L1 = 256
        L2 = 16  # sqrt(256)

        self.linear = nn.Linear(watermark_length, L1)
        self.L2 = L2

        # Conv to increase channels after upsampling
        self.conv = nn.Conv2d(1, embed_dim, kernel_size=3, padding=1)

    def forward(self, watermark, target_size):
        """
        Args:
            watermark: (B, L) - binary watermark bits
            target_size: (H, W) - size to match
        Returns:
            (B, C1, H, W) - spatially expanded watermark features
        """
        B = watermark.shape[0]

        # Linear projection
        x = self.linear(watermark)  # (B, L1)

        # Reshape to spatial
        x = x.view(B, 1, self.L2, self.L2)  # (B, 1, L2, L2)

        # Upsample to target size using nearest neighbor
        # From paper: "nearest-neighbor interpolation method is used"
        x = F.interpolate(x, size=target_size, mode='nearest')  # (B, 1, H, W)

        # Increase channels
        x = self.conv(x)  # (B, C1, H, W)

        return x


def window_partition(x, window_size):
    """
    Partition feature map into non-overlapping windows.

    Args:
        x: (B, C, H, W)
        window_size: Window size M
    Returns:
        windows: (B*num_windows, C, M, M)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous() # (B, num_windows_H, num_windows_W, C, M, M)

    # -1 is a cheesy syntax for saying "infer this dimension" 
    windows = windows.view(-1, C, window_size, window_size) # (B*num_windows, C, M, M)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Reverse window partition.

    Args:
        windows: (B*num_windows, C, M, M)
        window_size: Window size M
        H, W: Original height and width
    Returns:
        x: (B, C, H, W)
    """
    B_w, C, M, M = windows.shape
    B = B_w // ((H // window_size) * (W // window_size))
    x = windows.view(B, H // window_size, W // window_size, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(B, C, H, W)
    return x

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
    
