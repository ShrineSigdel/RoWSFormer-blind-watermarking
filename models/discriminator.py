import torch
import torch.nn as nn


class WatermarkLoss(nn.Module):
    """
    Combined loss for RoWSFormer training.
    
    Implements three loss components:
    1. Image fidelity (invisibility)
    2. Watermark recovery (robustness)
    3. Pixel constraint (validity)
    
    Args:
        lambda_1: Weight for image loss (default 2.0)
        lambda_2: Weight for watermark loss (default 10.0)
        lambda_3: Weight for constraint loss (default 0.1)
    """
    def __init__(self, lambda_1=2.0, lambda_2=10.0, lambda_3=0.1):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        
        self.mse = nn.MSELoss()
    
    def image_loss(self, cover, watermarked):
        """
        L_E: Image fidelity loss.
        Ensures watermark is imperceptible.
        
        Args:
            cover: Original image (B, 3, H, W)
            watermarked: Watermarked image (B, 3, H, W)
        """
        return self.mse(cover, watermarked)
    
    def watermark_loss(self, original_wm, extracted_wm):
        """
        L_D: Watermark recovery loss.
        Ensures watermark can be accurately extracted.
        
        Args:
            original_wm: Original watermark (B, L)
            extracted_wm: Extracted watermark (B, L)
        """
        return self.mse(original_wm, extracted_wm)
    
    def constraint_loss(self, watermarked):
        """
        L_C: Pixel constraint loss.
        Penalizes pixels outside [0, 1] range.
        
        From paper Eq. (6):
        L_C = Σ_{i,j} penalty(I_em[i,j])
        where penalty = 0.5 * |I - 1| if I > 1
                      = 0.5 * |I| if I < 0
                      = 0 otherwise
        
        Args:
            watermarked: Watermarked image (B, 3, H, W)
        """
        # Pixels > 1
        over = watermarked - 1.0
        over = torch.clamp(over, min=0)  # Only positive parts
        loss_over = 0.5 * over.abs().sum()
        
        # Pixels < 0
        under = watermarked
        under = torch.clamp(under, max=0)  # Only negative parts
        loss_under = 0.5 * under.abs().sum()
        
        # Normalize by number of pixels
        num_pixels = watermarked.numel()
        
        return (loss_over + loss_under) / num_pixels
    
    def forward(self, cover, watermarked, original_wm, extracted_wm):
        """
        Compute total loss.
        
        Args:
            cover: Original image (B, 3, H, W)
            watermarked: Watermarked image (B, 3, H, W)
            original_wm: Original watermark (B, L)
            extracted_wm: Extracted watermark (B, L)
        
        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Dictionary with individual loss values
        """
        # Compute individual losses
        L_E = self.image_loss(cover, watermarked)
        L_D = self.watermark_loss(original_wm, extracted_wm)
        L_C = self.constraint_loss(watermarked)
        
        # Total loss (Eq. 7)
        total_loss = self.lambda_1 * L_E + self.lambda_2 * L_D + self.lambda_3 * L_C
        
        # Return individual losses for logging
        loss_dict = {
            'total': total_loss.item(),
            'image': L_E.item(),
            'watermark': L_D.item(),
            'constraint': L_C.item()
        }
        
        return total_loss, loss_dict
