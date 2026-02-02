import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

class NoiseLayer(nn.Module):
    """
    Differentiable noise layer for training robustness.
    
    Implements various geometric and non-geometric attacks.
    All operations are differentiable to enable end-to-end training.
    
    Args:
        attack_types: List of attack types to use
    """
    def __init__(self, attack_types=None):
        super().__init__()
        
        if attack_types is None:
            # Default: use all attacks
            self.attack_types = [
                'none',  # Sometimes no attack
                'gaussian_noise',
                'gaussian_blur',
                'cropout',
                'dropout',
            ]
        else:
            self.attack_types = attack_types
    
    def gaussian_noise(self, x, sigma=None):
        """
        Add Gaussian noise.
        From paper: σ ∈ [0.001, 0.04] during training
        """
        if sigma is None:
            sigma = torch.rand(1).item() * 0.039 + 0.001
        
        noise = torch.randn_like(x) * sigma
        return torch.clamp(x + noise, 0, 1)
    
    def gaussian_blur(self, x, kernel_size=5):
        """
        Apply Gaussian blur.
        Approximated using average pooling for differentiability.
        """
        # Simple average blur (differentiable)
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        kernel = kernel.to(x.device)
        
        # Apply to each channel
        B, C, H, W = x.shape
        x_blur = []
        for c in range(C):
            blurred = F.conv2d(
                x[:, c:c+1], 
                kernel, 
                padding=kernel_size//2
            )
            x_blur.append(blurred)
        
        return torch.cat(x_blur, dim=1)
    
    def cropout(self, x, original, ratio=None):
        """
        Cropout attack: replace random region with original.
        From paper: ratio ∈ [0.1, 0.5] during training
        
        Args:
            x: Watermarked image
            original: Original cover image
            ratio: Fraction of image to crop
        """
        if ratio is None:
            ratio = torch.rand(1).item() * 0.4 + 0.1  # [0.1, 0.5]
        
        B, C, H, W = x.shape
        
        # Random crop size
        crop_h = int(H * ratio ** 0.5)
        crop_w = int(W * ratio ** 0.5)
        
        # Random position
        top = torch.randint(0, H - crop_h + 1, (1,)).item()
        left = torch.randint(0, W - crop_w + 1, (1,)).item()
        
        # Create mask
        mask = torch.ones_like(x)
        mask[:, :, top:top+crop_h, left:left+crop_w] = 0
        
        # Apply cropout
        return x * mask + original * (1 - mask)
    
    def dropout(self, x, original, ratio=None):
        """
        Dropout attack: replace random pixels with original.
        From paper: ratio ∈ [0.2, 0.6] during training
        
        Args:
            x: Watermarked image
            original: Original cover image
            ratio: Fraction of pixels to drop
        """
        if ratio is None:
            ratio = torch.rand(1).item() * 0.4 + 0.2  # [0.2, 0.6]
        
        # Random mask
        mask = (torch.rand_like(x) > ratio).float()
        
        # Apply dropout
        return x * mask + original * (1 - mask)
    
    def forward(self, x, original=None, attack_type=None):
        """
        Apply random attack.
        
        Args:
            x: Watermarked image (B, 3, H, W)
            original: Original image (needed for cropout/dropout)
            attack_type: Specific attack to use (random if None)
        Returns:
            attacked: (B, 3, H, W)
        """
        if attack_type is None:
            attack_type = np.random.choice(self.attack_types)
        
        if attack_type == 'none':
            return x
        elif attack_type == 'gaussian_noise':
            return self.gaussian_noise(x)
        elif attack_type == 'gaussian_blur':
            return self.gaussian_blur(x)
        elif attack_type == 'cropout':
            if original is None:
                return x
            return self.cropout(x, original)
        elif attack_type == 'dropout':
            if original is None:
                return x
            return self.dropout(x, original)
        else:
            return x
