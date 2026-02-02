import torch.nn as nn
from models.layers.embeddings import PatchEmbedding
from models.blocks.lcestb import LCESTB
from models.blocks.fetb import FETB

class Decoder(nn.Module):
    """
    RoWSFormer Decoder.
    
    Extracts watermark from (attacked) watermarked image.
    Blind extraction - no access to original image.
    
    Args:
        in_channels: Input channels (3 for RGB)
        base_dim: Base channel dimension (C)
        num_stages: Number of downsampling stages (K)
        watermark_length: Watermark bit length (64)
    """
    def __init__(self, in_channels=3, base_dim=64, num_stages=3, watermark_length=64):
        super().__init__()
        self.num_stages = num_stages
        self.base_dim = base_dim
        self.watermark_length = watermark_length
        
        # Initial feature extraction
        # From paper: "we also use a 3×3 convolution to extract 
        # the shallow features"
        self.patch_embed = PatchEmbedding(in_channels, base_dim)
        
        # Downsampling stages with LCESTB
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        for i in range(num_stages):
            dim = base_dim * (2 ** i)
            
            # LCESTB blocks (W-MSA and SW-MSA)
            self.down_blocks.append(
                nn.Sequential(
                    LCESTB(dim, num_heads=4, window_size=8, shift_size=0),
                    LCESTB(dim, num_heads=4, window_size=8, shift_size=4)
                )
            )
            
            # Downsampling
            if i < num_stages - 1:
                self.down_samples.append(
                    nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1)
                )
        
        # Bottleneck with FETB
        bottleneck_dim = base_dim * (2 ** (num_stages - 1))
        self.bottleneck = FETB(bottleneck_dim, num_heads=4, num_blocks=2)
        
        # Information extraction layer
        # From paper: "I_output_no is passed through the information 
        # extraction layer, consisting of a convolutional layer and 
        # a fully connected layer"
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # FC layers for watermark extraction
        self.fc = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim // 2, watermark_length),
            nn.Sigmoid()  # Output in [0, 1] for binary classification
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) - (attacked) watermarked image
        Returns:
            watermark: (B, L) - extracted watermark bits (continuous [0,1])
        """
        B, C, H, W = x.shape
        current_h, current_w = H, W
        
        # Initial features
        x = self.patch_embed(x)  # (B, base_dim, H, W)
        
        # Downsampling with LCESTB
        for i, (block, downsample) in enumerate(
            zip(self.down_blocks, self.down_samples + [None])
        ):
            x = block[0](x, current_h, current_w)
            x = block[1](x, current_h, current_w)
            
            if downsample is not None:
                x = downsample(x)
                current_h, current_w = current_h // 2, current_w // 2
        
        # Bottleneck
        x = self.bottleneck(x)  # (B, bottleneck_dim, H/2^K, W/2^K)
        
        # Global pooling
        x = self.pool(x)  # (B, bottleneck_dim, 1, 1)
        x = x.view(B, -1)  # (B, bottleneck_dim)
        
        # Extract watermark
        watermark = self.fc(x)  # (B, watermark_length)
        
        return watermark
