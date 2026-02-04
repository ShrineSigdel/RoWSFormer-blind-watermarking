import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

from utils.dataset import RoWSFormerDataset
from utils.metrics import compute_psnr, compute_bit_accuracy
from models.encoder import Encoder
from models.decoder import Decoder
from models.discriminator import WatermarkLoss
from models.layers.noise import NoiseLayer
from train import train


def save_model(encoder, decoder, optimizer, epoch, path='checkpoints/model.pth'):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"✓ Model saved to {path}")


def load_model(encoder, decoder, optimizer=None, path='checkpoints/model.pth'):
    """Load model checkpoint."""
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return 0
    
    checkpoint = torch.load(path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f"✓ Model loaded from {path} (epoch {epoch})")
    return epoch


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ============ Configuration ============
    # ============ Configuration ============
    batch_size = 4
    num_epochs = 50             # More epochs
    learning_rate = 2e-4
    base_dim = 64               # Larger model
    num_stages = 3
    watermark_length = 64
    checkpoint_path = 'checkpoints/model.pth'
    
    # ============ Data ============
    train_hr_dir = './data/DIV2K_train_HR/DIV2K_train_HR/'
    
    train_dataset = RoWSFormerDataset(train_hr_dir, img_size=128, bit_length=watermark_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    print(f"Dataset size: {len(train_dataset)} images")

    # ============ Models ============
    encoder = Encoder(
        in_channels=3,
        base_dim=base_dim,
        num_stages=num_stages,
        watermark_length=watermark_length
    ).to(device)
    
    decoder = Decoder(
        in_channels=3,
        base_dim=base_dim,
        num_stages=num_stages,
        watermark_length=watermark_length
    ).to(device)
    
    noise_layer = NoiseLayer()
    
    # ============ Loss & Optimizer ============
    criterion = WatermarkLoss(lambda_1=2.0, lambda_2=10.0, lambda_3=0.1)
    
    # Combine encoder and decoder parameters
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=1e-4)
    
    # Print model info
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    print(f"Total parameters: {encoder_params + decoder_params:,}")
    
    # ============ Training ============
    print("\n" + "="*50)
    print("Starting Training...")
    print("="*50)
    
    train(encoder, decoder, noise_layer, train_loader,
          criterion, optimizer, device, num_epochs=num_epochs)
    
    # ============ Save Model ============
    save_model(encoder, decoder, optimizer, num_epochs, checkpoint_path)
    
    # ============ Test & Visualize ============
    print("\n" + "="*50)
    print("Testing on sample images...")
    print("="*50)
    
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        # Get a batch
        images, watermarks = next(iter(train_loader))
        images = images.to(device)
        watermarks = watermarks.to(device)
        
        # Encode
        watermarked = encoder(images, watermarks)
        
        # Decode (no attack)
        extracted = decoder(watermarked)
        
        # Compute metrics
        psnr = compute_psnr(images, watermarked)
        accuracy = compute_bit_accuracy(watermarks, extracted)
        
        print(f"\nResults (no attack):")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  Bit Accuracy: {accuracy:.2f}%")
        
        # Test with attack
        attacked = noise_layer(watermarked, images, attack_type='gaussian_noise')
        extracted_attacked = decoder(attacked)
        accuracy_attacked = compute_bit_accuracy(watermarks, extracted_attacked)
        
        print(f"\nResults (Gaussian noise attack):")
        print(f"  Bit Accuracy: {accuracy_attacked:.2f}%")
        
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()