import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable
from src.utils import setup_logging

logger = setup_logging("FashionAI.DataLoader")

class FashionDataset(Dataset):
    """Custom Dataset for FashionAI outfit images and labels."""
    def __init__(self, data_dir: str, labels_file: Optional[str] = None, transform: Optional[Callable] = None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        if labels_file and os.path.exists(labels_file):
            df = pd.read_csv(labels_file)
            for _, row in df.iterrows():
                img_path = os.path.join(data_dir, row['filename'])
                if os.path.exists(img_path):
                    self.samples.append((img_path, int(row['label'])))
        else:
            # Fallback to ImageFolder structure if no csv provided
            if os.path.exists(data_dir):
                classes = sorted(os.listdir(data_dir))
                self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes) if os.path.isdir(os.path.join(data_dir, cls_name))}
                for cls_name, idx in self.class_to_idx.items():
                    cls_dir = os.path.join(data_dir, cls_name)
                    for img_name in os.listdir(cls_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                            self.samples.append((os.path.join(cls_dir, img_name), idx))
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='gray')
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def get_data_loaders(config: dict, transforms_dict: dict) -> tuple:
    """Creates and returns training, validation, and test DataLoaders."""
    train_dataset = FashionDataset(
        data_dir=config['data']['train_dir'],
        labels_file=config['data']['labels_file'],
        transform=transforms_dict['train']
    )
    val_dataset = FashionDataset(
        data_dir=config['data']['val_dir'],
        transform=transforms_dict['val']
    )
    test_dataset = FashionDataset(
        data_dir=config['data']['test_dir'],
        transform=transforms_dict['test']
    )

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'], pin_memory=True)

    return train_loader, val_loader, test_loader