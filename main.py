import torch
from torch.utils.data import DataLoader

from utils.dataset import RoWSFormerDataset
from models.encoder import Encoder


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Define paths to training and validation high-resolution image directories
    train_hr_dir = './data/DIV2K_train_HR/DIV2K_train_HR/'
    val_hr_dir = './data/DIV2K_valid_HR/DIV2K_valid_HR/'
    
    # 2. Create dataset
    train_dataset = RoWSFormerDataset(train_hr_dir)
    valid_dataset = RoWSFormerDataset(val_hr_dir)

    # 3. Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    # 4. Initialize Encoder
    encoder = Encoder(
        in_channels=3,
        base_dim=64,
        num_stages=3,
        watermark_length=64
    ).to(device)
    
    print(f"Encoder initialized with {sum(p.numel() for p in encoder.parameters()):,} parameters")

    # 5. Process one batch for testing
    for batch_idx, batch in enumerate(train_loader):
        images, watermarks = batch
        images = images.to(device)
        watermarks = watermarks.to(device)
        
        print(f"\n=== Processing Batch {batch_idx + 1} ===")
        print(f"Cover image shape: {images.shape}")
        print(f"Watermark shape: {watermarks.shape}")
        print(f"Cover image range: [{images.min().item():.3f}, {images.max().item():.3f}]")
        
        # Encode watermark into image
        with torch.no_grad():  # No gradients needed for inference
            watermarked_images = encoder(images, watermarks)
        
        print(f"Watermarked image shape: {watermarked_images.shape}")
        print(f"Watermarked image range: [{watermarked_images.min().item():.3f}, {watermarked_images.max().item():.3f}]")
        
        # Calculate perturbation statistics
        perturbation = watermarked_images - images
        print(f"\nPerturbation statistics:")
        print(f"  Mean: {perturbation.mean().item():.6f}")
        print(f"  Std: {perturbation.std().item():.6f}")
        print(f"  Min: {perturbation.min().item():.6f}")
        print(f"  Max: {perturbation.max().item():.6f}")
        
        # Calculate PSNR (Peak Signal-to-Noise Ratio)
        mse = torch.mean((watermarked_images - images) ** 2)
        psnr = 10 * torch.log10(1.0 / mse)
        print(f"  PSNR: {psnr.item():.2f} dB")
        
        break  # Process only one batch for testing

    print("\n✓ Encoding test completed successfully!")


if __name__ == "__main__":
    main()