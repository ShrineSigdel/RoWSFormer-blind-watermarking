import torch.nn as nn

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
