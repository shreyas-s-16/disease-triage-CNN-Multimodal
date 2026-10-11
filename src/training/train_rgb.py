"""
Training script for RGB baseline model.
"""
import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path

from src.utils import get_device, set_seed, project_root
from src.data_loaders import create_rgb_dataloaders
from src.models import RGBEncoder


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
    
    return running_loss / total, 100. * correct / total


def validate(model, loader, criterion, device, epoch):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
    
    return running_loss / total, 100. * correct / total


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    set_seed(config.get('seed', 42))
    device = get_device()
    
    # Data
    train_loader, val_loader, info = create_rgb_dataloaders(
        metadata_path=config['data']['metadata_path'],
        images_dir=config['data']['images_dir'],
        batch_size=config['training']['batch_size'],
        image_size=config['data']['image_size'],
        subset_frac=config['data'].get('subset_frac', 1.0),
        balance_classes=config['data'].get('balance_classes', True),
        val_split=config['data'].get('val_split', 0.2),
        num_workers=config['training'].get('num_workers', 4),
        random_seed=config.get('seed', 42)
    )
    
    # Model
    model = RGBEncoder(
        pretrained=config['model']['pretrained'],
        frozen=config['model']['frozen'],
        output_dim=config['model']['output_dim'],
        num_classes=info['num_classes']
    ).to(device)
    
    # Loss with class weights
    class_weights = info['class_weights'].to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer (only train projection + classifier if frozen)
    if config['model']['frozen']:
        params = list(model.projection.parameters()) + list(model.classifier.parameters())
    else:
        params = model.parameters()
    
    optimizer = torch.optim.AdamW(
        params,
        lr=float(config['training']['lr']),
        weight_decay=float(config['training'].get('weight_decay', 1e-4))
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs']
    )
    
    # Checkpoint dir
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(1, config['training']['epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config,
                'class_names': info['class_names']
            }, checkpoint_dir / 'best_model.pt')
            print(f"  -> Saved best model (val_acc={val_acc:.2f}%)")
        
        # Save checkpoint
        if epoch % config['training'].get('save_every', 5) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
    
    # Save final model
    torch.save({
        'epoch': config['training']['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'config': config,
        'class_names': info['class_names']
    }, checkpoint_dir / 'final_model.pt')
    
    # Save history
    np.save(checkpoint_dir / 'history.npy', history)
    
    print(f"\nTraining complete. Best val acc: {best_val_acc:.2f}%")
    print(f"Model saved to {checkpoint_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()
    main(args.config)