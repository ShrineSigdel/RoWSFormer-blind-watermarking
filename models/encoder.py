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

class LocallyChannelEnhancedBlock(nn.Module):
    """
    Locally-Channel Enhanced Block (LCEB).

    Captures local features with depthwise conv and channel attention.

    From paper:
    1. Linear projection (expand)
    2. Depthwise 3×3 conv (local features)
    3. Linear projection (compress)
    4. Channel attention (reweight channels)

    Args:
        dim: Channel dimension
        expansion: Channel expansion ratio (default 2)
    """
    def __init__(self, dim, expansion=2):
        super().__init__()
        hidden_dim = int(dim * expansion)

        # Linear projection to expand channels
        self.expand = nn.Conv2d(dim, hidden_dim, kernel_size=1)

        # Depthwise convolution for local features
        # From paper: "3×3 depthwise convolution to capture local features"
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                               padding=1, groups=hidden_dim)

        # Linear projection to compress back
        self.compress = nn.Conv2d(hidden_dim, dim, kernel_size=1)

        # Channel attention
        # From paper: "a pooling layer followed by a fully connected layer
        # is used to compute attention weights for each channel"
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

        self.act = nn.GELU()

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            bias: (B, C, H, W) - local and channel-enhanced features
        """
        identity = x

        # Expand
        x = self.expand(x)
        x = self.act(x)

        # Depthwise conv for local features
        x = self.dwconv(x)
        x = self.act(x)

        # Compress
        x = self.compress(x)

        # Channel attention
        B, C, H, W = x.shape
        channel_weights = self.pool(x).view(B, C)  # (B, C)
        channel_weights = self.channel_attn(channel_weights).view(B, C, 1, 1)  # (B, C, 1, 1)

        # Apply channel attention
        x = x * channel_weights

        # Add to identity (residual)
        return identity + x
    
class LCESTB(nn.Module):
    """
    Locally-Channel Enhanced Swin Transformer Block.

    Combines:
    1. Swin Transformer Block (global context via windowed attention)
    2. Locally-Channel Enhanced Block (local features + channel attention)

    This is the core building block of RoWSFormer encoder and decoder.

    Args:
        dim: Channel dimension
        num_heads: Number of attention heads
        window_size: Window size for attention
        shift_size: Whether to use shifted windows (0 or window_size//2)
    """
    def __init__(self, dim, num_heads=4, window_size=8, shift_size=0):
        super().__init__()

        # From paper: "LESTB consists of two main parts"
        # Part 1: Swin Transformer Block
        self.swin_block = SwinTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size
        )

        # Part 2: Locally-Channel Enhanced Block
        self.lce_block = LocallyChannelEnhancedBlock(dim=dim)

    def forward(self, x, H, W):
        """
        Args:
            x: (B, C, H, W)
            H, W: Spatial dimensions
        Returns:
            (B, C, H, W)
        """
        # First: Swin block for global modeling
        x = self.swin_block(x, H, W)

        # Second: LCE block for local and channel modeling
        x = self.lce_block(x)

        return x

class FrequencyEnhancedBlock(nn.Module):
    """
    Frequency-Enhanced Block using simplified frequency modeling.

    From paper:
    - Applies DCT to extract frequency features
    - Uses FC layer to compute frequency attention weights

    Simplified implementation:
    - Uses learnable frequency filters instead of explicit DCT
    - Applies channel-wise frequency attention

    Args:
        dim: Channel dimension
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Global pooling to get frequency statistics
        self.pool = nn.AdaptiveAvgPool2d(1)

        # FC layer to compute frequency attention weights
        # From paper: "a simple fully connected (FC) layer is used to
        # compute the frequency domain attention weights"
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W) - frequency-enhanced features
        """
        B, C, H, W = x.shape

        # Global pooling (approximates frequency spectrum statistics)
        freq_stats = self.pool(x).view(B, C)  # (B, C)

        # Compute frequency attention weights
        freq_weights = self.fc(freq_stats).view(B, C, 1, 1)  # (B, C, 1, 1)

        # Apply frequency attention
        out = x * freq_weights

        return out

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

class FETB(nn.Module):
    """
    Frequency-Enhanced Transformer Block.

    Combines:
    1. Global Transformer Block (capture global context)
    2. Frequency-Enhanced Block (frequency domain modeling)

    Used at the bottleneck of the encoder/decoder.

    Args:
        dim: Channel dimension
        num_heads: Number of attention heads
        num_blocks: Number of Transformer blocks (default 2)
    """
    def __init__(self, dim, num_heads=4, num_blocks=2):
        super().__init__()

        # Multiple Transformer blocks
        self.blocks = nn.ModuleList([
            GlobalTransformerBlock(dim, num_heads)
            for _ in range(num_blocks)
        ])

        # Frequency enhancement
        self.freq_enhance = FrequencyEnhancedBlock(dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Apply frequency enhancement
        x = self.freq_enhance(x)

        return x
