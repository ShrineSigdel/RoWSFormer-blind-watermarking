import torch
from torch.utils.data import DataLoader

from utils.dataset import RoWSFormerDataset
from utils.metrics import compute_psnr, compute_bit_accuracy
from models.encoder import Encoder
from models.decoder import Decoder
from models.layers.noise import NoiseLayer
from main import load_model


def test():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ============ Configuration ============
    base_dim = 32
    num_stages = 2
    watermark_length = 64
    checkpoint_path = 'checkpoints/model.pth'
    
    # ============ Data ============
    test_hr_dir = './data/DIV2K_valid_HR/DIV2K_valid_HR/'
    test_dataset = RoWSFormerDataset(test_hr_dir, img_size=128, bit_length=watermark_length)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    print(f"Test dataset size: {len(test_dataset)} images")

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
    
    # ============ Load Model ============
    load_model(encoder, decoder, path=checkpoint_path)
    
    # ============ Test ============
    encoder.eval()
    decoder.eval()
    
    attack_types = ['none', 'gaussian_noise', 'gaussian_blur', 'cropout', 'dropout']
    
    print("\n" + "="*50)
    print("Running Tests...")
    print("="*50)
    
    with torch.no_grad():
        images, watermarks = next(iter(test_loader))
        images = images.to(device)
        watermarks = watermarks.to(device)
        
        # Encode
        watermarked = encoder(images, watermarks)
        psnr = compute_psnr(images, watermarked)
        print(f"\nPSNR: {psnr:.2f} dB")
        
        # Test each attack
        print("\nBit Accuracy by Attack:")
        for attack in attack_types:
            attacked = noise_layer(watermarked, images, attack_type=attack)
            extracted = decoder(attacked)
            accuracy = compute_bit_accuracy(watermarks, extracted)
            print(f"  {attack:20s}: {accuracy:.2f}%")
    
    print("\n✓ Testing complete!")


if __name__ == "__main__":
    test()