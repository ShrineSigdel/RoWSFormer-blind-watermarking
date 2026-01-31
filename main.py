from torch.utils.data import DataLoader


from utils.dataset import RoWSFormerDataset
from models.encoder import PatchEmbedding, WatermarkEncoder


def main():

    # 1. Define paths to training and validation high-resolution image directories
    train_hr_dir = './data/DIV2K_train_HR/DIV2K_train_HR/'  # Path to training high-resolution images
    val_hr_dir = './data/DIV2K_valid_HR/DIV2K_valid_HR/'    # Path to validation high-resolution images
    
    # 2. Create two separate dataset objects
    train_dataset = RoWSFormerDataset(train_hr_dir)
    valid_dataset = RoWSFormerDataset(val_hr_dir)

    # 3. Create two separate DataLoaders
    # We shuffle training data to improve learning, but usually don't need to shuffle validation
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)


    # 4. Iterate through one batch of training data and perform encoding for testing purpose
    for batch in train_loader:
        images, watermarks = batch  # Unpack the batch
        print(f"Image batch shape: {images.shape}")          
        print(f"Watermark batch shape: {watermarks.shape}")  

        # Initialize encoders
        patch_embedder = PatchEmbedding(in_channels=3, embed_dim=64, patch_size=4)
        wm_encoder = WatermarkEncoder(watermark_length=64, embed_dim=32, image_size=images.shape[2])

        # # Encode images and watermarks
        image_features = patch_embedder(images)  # (B, C, H, W)
        wm_features = wm_encoder(watermarks, target_size=(images.shape[2], images.shape[3]))  # (B, C1, H, W)

        print(f"Encoded image features shape: {image_features.shape}")
        print(f"Encoded watermark features shape: {wm_features.shape}")

        break  # Just process one batch for this test

if __name__ == "__main__":
    main()