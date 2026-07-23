import torchvision.transforms as T
from typing import Dict

def get_transforms(image_size: int = 224) -> Dict[str, T.Compose]:
    """Returns training, validation, and inference augmentation pipelines."""
    train_transform = T.Compose([
        T.Resize((int(image_size * 1.15), int(image_size * 1.15))),
        T.RandomCrop((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return {
        'train': train_transform,
        'val': val_test_transform,
        'test': val_test_transform
    }