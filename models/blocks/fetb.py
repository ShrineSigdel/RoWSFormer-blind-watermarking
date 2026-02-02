
import torch
import torch.nn as nn

from models.blocks.frequency import FrequencyEnhancedBlock
from models.blocks.global_transformer import GlobalTransformerBlock

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
