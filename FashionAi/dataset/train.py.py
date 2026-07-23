import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from src.utils import setup_logging, load_config, set_seed
from src.model import FashionClassifier
from src.data_loader import get_data_loaders
from src.augmentations import get_transforms

logger = setup_logging("FashionAI.Train")

def train_model():
    config = load_config()
    set_seed(config['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")
    
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    writer = SummaryWriter(log_dir=config['training']['log_dir'])
    
    transforms = get_transforms(config['data']['image_size'])
    train_loader, val_loader, _ = get_data_loaders(config, transforms)
    
    model = FashionClassifier(
        backbone=config['model']['backbone'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        dropout_rate=config['model']['dropout_rate']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    scaler = GradScaler(enabled=config['training']['mixed_precision'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config['training']['epochs']}]")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with autocast(enabled=config['training']['mixed_precision']):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            train_total += labels.size(0)
            train_correct += preds.eq(labels).sum().item()
            
            loop.set_postfix(loss=loss.item())
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # Validation Loop
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        scheduler.step(epoch_val_loss)
        
        writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/Val', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/Train', epoch_train_acc, epoch)
        writer.add_scalar('Accuracy/Val', epoch_val_acc, epoch)
        
        logger.info(f"Epoch {epoch+1} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            ckpt_path = os.path.join(config['training']['checkpoint_dir'], 'best_model.pth')
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved best model checkpoint to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= config['training']['patience']:
                logger.info("Early stopping triggered.")
                break
                
    writer.close()

if __name__ == '__main__':
    train_model()