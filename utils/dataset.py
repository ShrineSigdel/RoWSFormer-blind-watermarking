import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class RoWSFormerDataset(Dataset):

    def __init__(self, img_dir, img_size=128, bit_length=64):


        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.bit_length = bit_length
        
        # Standard Pre-processing for RoWSFormer
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)), # Resize to 128x128
            transforms.ToTensor(),                  # Convert to 0.0 - 1.0 range
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. Load Image
        image = Image.open(self.img_paths[idx]).convert('RGB')
        image = self.transform(image)
        
        # 2. Generate the random 64-bit Watermark (0s and 1s)
        watermark = torch.randint(0, 2, (self.bit_length,)).float()
        
        return image, watermark
    