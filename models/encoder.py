import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.embeddings import PatchEmbedding
from models.layers.watermark_encoder import WatermarkEncoder
from models.blocks.lcestb import LCESTB
from models.blocks.fetb import FETB

class Encoder(nn.Module):
    """
    RoWSFormer Encoder.

    Embeds watermark into cover image using U-Net architecture with:
    - LCESTB blocks for feature extraction
    - FETB at bottleneck
    - Skip connections
    - Watermark feature fusion
"
    Args:
        in_channels: Input channels (3 for RGB)
        base_dim: Base channel dimension (C)
        num_stages: Number of down/up sampling stages (K)
        watermark_length: Watermark bit length (64)
    """
    def __init__(self, in_channels=3, base_dim=64, num_stages=3, watermark_length=64):
        super().__init__()
        self.num_stages = num_stages
        self.base_dim = base_dim

        # Initial feature extraction
        self.patch_embed = PatchEmbedding(in_channels, base_dim)

        # Watermark encoder
        self.watermark_encoder = WatermarkEncoder(
            watermark_length=watermark_length,
            embed_dim=base_dim // 2  # C1 in paper
        )

        # Downsampling stages
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        for i in range(num_stages):
            dim = base_dim * (2 ** i)

            # LCESTB (alternating between W-MSA and SW-MSA)
            self.down_blocks.append(
                nn.Sequential(
                    LCESTB(dim, num_heads=4, window_size=8, shift_size=0),
                    LCESTB(dim, num_heads=4, window_size=8, shift_size=4)
                )
            )

            # Downsampling (except last stage)
            if i < num_stages - 1:
                self.down_samples.append(
                    nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1)
                )

        # Bottleneck with FETB
        bottleneck_dim = base_dim * (2 ** (num_stages - 1))
        self.bottleneck = FETB(bottleneck_dim, num_heads=4, num_blocks=2)

        # Upsampling stages
        self.up_samples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for i in range(num_stages - 1, -1, -1):
            dim = base_dim * (2 ** i)

            # Upsampling (except first upsampling stage)
            if i < num_stages - 1:
                self.up_samples.append(
                    nn.ConvTranspose2d(dim * 2, dim, kernel_size=2, stride=2)
                )

            # Concatenation: upsampled + skip + watermark
            # From paper: concatenated with feature map from downsampling
            # and watermark feature map
            concat_dim = dim * 2 + base_dim // 2 if i < num_stages - 1 else dim

            self.up_blocks.append(
                nn.Sequential(
                    nn.Conv2d(concat_dim, dim, kernel_size=3, padding=1),
                    LCESTB(dim, num_heads=4, window_size=8, shift_size=0),
                    LCESTB(dim, num_heads=4, window_size=8, shift_size=4)
                )
            )

        # Output layer
        self.output = nn.Conv2d(base_dim, in_channels, kernel_size=3, padding=1)

    def forward(self, image, watermark):
        """
        Args:
            image: (B, 3, H, W) - cover image
            watermark: (B, L) - watermark bits
        Returns:
            watermarked: (B, 3, H, W) - watermarked image
        """
        B, C, H, W = image.shape

        # Initial features
        x = self.patch_embed(image)  # (B, C, H, W)

        # Downsampling
        skip_connections = []
        current_h, current_w = H, W

        for i, (block, downsample) in enumerate(
            zip(self.down_blocks, self.down_samples + [None])
        ):
            x = block[0](x, current_h, current_w)
            x = block[1](x, current_h, current_w)
            skip_connections.append(x)

            if downsample is not None:
                x = downsample(x)
                current_h, current_w = current_h // 2, current_w // 2

        # Bottleneck
        x = self.bottleneck(x)

        # Upsampling
        skip_connections = skip_connections[::-1]  # Reverse

        for i, (upsample, block) in enumerate(
            zip([None] + list(self.up_samples), self.up_blocks)
        ):
            if upsample is not None:
                x = upsample(x)
                current_h, current_w = current_h * 2, current_w * 2

            # Get watermark features for this resolution
            wm_features = self.watermark_encoder(
                watermark,
                target_size=(current_h, current_w)
            )

            # Concatenate: upsampled + skip + watermark
            if i > 0:
                x = torch.cat([x, skip_connections[i], wm_features], dim=1)
            else:
                x = skip_connections[i]  # First iteration

            x = block[0](x)  # Conv
            x = block[1](x, current_h, current_w)  # LCESTB 1
            x = block[2](x, current_h, current_w)  # LCESTB 2

        # Output (perturbation ΔI)
        delta = self.output(x)  # (B, 3, H, W)

        # Residual: I_em = I_co + ΔI
        watermarked = image + delta

        # Clamp to [0, 1]
        watermarked = torch.clamp(watermarked, 0, 1)

        return watermarked
