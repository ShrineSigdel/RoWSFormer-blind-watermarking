import torch.nn as nn

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
   