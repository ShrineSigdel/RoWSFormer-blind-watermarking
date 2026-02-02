import torch
import torch.nn as nn

from models.blocks.swin_block import SwinTransformerBlock
from models.blocks.lceb import LocallyChannelEnhancedBlock

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
