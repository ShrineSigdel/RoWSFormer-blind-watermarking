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

