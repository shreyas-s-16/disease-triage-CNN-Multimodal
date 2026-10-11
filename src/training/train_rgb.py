"""
Training script for RGB baseline model.
Supports stratified 70/15/15 splits, early stopping, and resumable checkpoints.
"""
import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import get_device, set_seed
from src.data_loaders import HAM10000Dataset, get_rgb_transforms
from src.models import RGBEncoder


def create_stratified_splits(dataset: HAM10000Dataset, splits_path: str, seed: int = 42):
    """Create stratified 70/15/15 train/val/test splits and save to CSV."""
    labels = dataset.df['label'].values
    indices = np.arange(len(dataset))
    
    # First split: 70% train, 30% temp
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, stratify=labels, random_state=seed
    )
    
    # Second split: 15% val, 15% test from temp
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=seed
    )
    
    # Save splits
    splits_df = pd.DataFrame({
        'split': ['train'] * len(train_idx) + ['val'] * len(val_idx) + ['test'] * len(test_idx),
        'index': list(train_idx) + list(val_idx) + list(test_idx)
    })
    splits_df.to_csv(splits_path, index=False)
    print(f"Saved splits to {splits_path}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    return train_idx, val_idx, test_idx


def load_splits(splits_path: str):
    """Load existing splits from CSV."""
    splits_df = pd.read_csv(splits_path)
    train_idx = splits_df[splits_df['split'] == 'train']['index'].values
    val_idx = splits_df[splits_df['split'] == 'val']['index'].values
    test_idx = splits_df[splits_df['split'] == 'test']['index'].values
    return train_idx, val_idx, test_idx


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
    
    # Create full dataset
    full_dataset = HAM10000Dataset(
        metadata_path=config['data']['metadata_path'],
        images_dir=config['data']['images_dir'],
        transform=None,
        subset_frac=config['data'].get('subset_frac', 1.0),
        random_seed=config.get('seed', 42),
        balance_classes=False
    )
    
    # Create or load stratified splits
    splits_path = Path(config['data'].get('splits_path', 'data/rgb/processed/splits.csv'))
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    
    if splits_path.exists():
        train_idx, val_idx, test_idx = load_splits(str(splits_path))
        print(f"Loaded existing splits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    else:
        train_idx, val_idx, test_idx = create_stratified_splits(full_dataset, str(splits_path), config.get('seed', 42))
    
    # Create subset datasets with appropriate transforms
    train_transform = get_rgb_transforms(config['data']['image_size'], is_train=True)
    val_transform = get_rgb_transforms(config['data']['image_size'], is_train=False)
    
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Sampler for balanced training
    if config['data'].get('balance_classes', True):
        train_labels = [full_dataset.df.iloc[i]['label'] for i in train_idx]
        class_weights = full_dataset.get_class_weights()
        sample_weights = [class_weights[label] for label in train_labels]
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    else:
        sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    info = {
        'num_classes': len(HAM10000Dataset.CLASSES),
        'class_names': HAM10000Dataset.CLASSES,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'class_weights': full_dataset.get_class_weights()
    }
    
    # Model
    model = RGBEncoder(
        pretrained=config['model']['pretrained'],
        frozen=config['model']['frozen'],
        output_dim=config['model']['output_dim'],
        num_classes=info['num_classes']
    ).to(device)
    
    # Resume from checkpoint if exists
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_path = checkpoint_dir / 'checkpoint_last.pt'
    start_epoch = 1
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    if resume_path.exists():
        print(f"Resuming from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)
        history = checkpoint.get('history', history)
    
    # Loss with class weights
    class_weights = info['class_weights'].to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    if config['model']['frozen']:
        params = list(model.projection.parameters()) + list(model.classifier.parameters())
    else:
        params = model.parameters()
    
    optimizer = torch.optim.AdamW(
        params,
        lr=float(config['training']['lr']),
        weight_decay=float(config['training'].get('weight_decay', 1e-4))
    )
    
    # Resume optimizer state if available
    if resume_path.exists() and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs']
    )
    
    if resume_path.exists() and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # Step scheduler to correct epoch
        for _ in range(start_epoch - 1):
            scheduler.step()
    
    # Early stopping config
    early_stopping_patience = config['training'].get('early_stopping_patience', 5)
    early_stopping_min_delta = config['training'].get('early_stopping_min_delta', 1e-4)
    
    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        # Save best model (by val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss,
                'config': config,
                'class_names': info['class_names'],
                'history': history
            }, checkpoint_dir / 'best_model.pt')
            print(f"  -> Saved best model (val_acc={val_acc:.2f}%)")
        
        # Save last checkpoint (for resuming)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'config': config,
            'history': history
        }, checkpoint_dir / 'checkpoint_last.pt')
        
        # Early stopping check
        if val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs (patience={early_stopping_patience})")
                break
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'best_val_acc': best_val_acc,
        'config': config,
        'class_names': info['class_names'],
        'history': history
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