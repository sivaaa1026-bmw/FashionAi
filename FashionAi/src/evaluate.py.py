import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from src.utils import setup_logging, load_config
from src.model import FashionClassifier
from src.data_loader import get_data_loaders
from src.augmentations import get_transforms

logger = setup_logging("FashionAI.Evaluate")

def evaluate_model():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transforms = get_transforms(config['data']['image_size'])
    _, _, test_loader = get_data_loaders(config, transforms)
    
    model = FashionClassifier(
        backbone=config['model']['backbone'],
        num_classes=config['model']['num_classes']
    ).to(device)
    
    ckpt_path = os.path.join(config['training']['checkpoint_dir'], 'best_model.pth')
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        logger.info(f"Loaded checkpoint from {ckpt_path}")
    else:
        logger.warning("No checkpoint found! Evaluating with random weights.")
        
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    logger.info(f"Evaluation Results -> Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(classification_report(all_labels, all_preds, zero_division=0))

if __name__ == '__main__':
    import os
    evaluate_model()