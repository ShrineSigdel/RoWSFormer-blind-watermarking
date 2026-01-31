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
