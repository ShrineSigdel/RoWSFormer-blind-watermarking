import torch


def compute_psnr(img1, img2):
    """
    Compute PSNR between two images.
    
    Args:
        img1, img2: Images in range [0, 1], shape (B, C, H, W)
    
    Returns:
        psnr: PSNR value in dB (averaged over batch)
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 1.0
    psnr = 10 * torch.log10(max_pixel ** 2 / mse)
    return psnr.item()

def compute_bit_accuracy(original, extracted, threshold=0.5):
    """
    Compute bit extraction accuracy.
    
    Args:
        original: Original watermark bits (B, L), values in {0, 1}
        extracted: Extracted watermark (B, L), values in [0, 1]
        threshold: Threshold for binary classification (default 0.5)
    
    Returns:
        accuracy: Percentage of correctly extracted bits
    """
    # Threshold extracted watermark to binary
    extracted_binary = (extracted > threshold).float()
    
    # Compute accuracy
    correct = (extracted_binary == original).float()
    accuracy = correct.mean().item() * 100
    
    return accuracy
