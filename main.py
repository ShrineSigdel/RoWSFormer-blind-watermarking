from torch.utils.data import DataLoader
from utils.dataset import RoWSFormerDataset


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

    print(f"Training images: {len(train_dataset)}")   # Should be 800
    print(f"Validation images: {len(valid_dataset)}") # Should be 100

if __name__ == "__main__":
    main()