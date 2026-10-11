"""
Training script for Fusion model.
"""
import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

from src.utils import get_device, set_seed
from src.data_loaders import (
    create_rgb_dataloaders, create_tabular_dataloaders, create_thermal_dataloaders
)
from src.models import (
    RGBEncoder, ThermalEncoder, TabularEncoder, FusionEncoder, FusionModel
)


class FusionDataset(Dataset):
    """Combined dataset for multi-modal fusion training."""
    
    def __init__(
        self,
        rgb_dataset,
        thermal_dataset,
        tabular_dataset,
        split: str = 'train'
    ):
        self.rgb_dataset = rgb_dataset
        self.thermal_dataset = thermal_dataset
        self.tabular_dataset = tabular_dataset
        self.split = split
        
        # Use the minimum length to ensure all modalities have data
        self.length = min(len(rgb_dataset), len(thermal_dataset), len(tabular_dataset))
        print(f"FusionDataset {split}: {self.length} samples (min of RGB={len(rgb_dataset)}, "
              f"Thermal={len(thermal_dataset)}, Tabular={len(tabular_dataset)})")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        rgb_img, rgb_label = self.rgb_dataset[idx]
        thermal_img, _ = self.thermal_dataset[idx]
        tabular_data = self.tabular_dataset[idx]
        
        return {
            'rgb': rgb_img,
            'thermal': thermal_img,
            'tabular': tabular_data['symptoms'],
            'visual_label': rgb_label,
            'fever_label': tabular_data['fever'],
            'triage_label': tabular_data['triage']
        }


def create_fusion_dataloaders(
    rgb_metadata_path: str,
    rgb_images_dir: str,
    thermal_root_dir: str,
    tabular_dataset_path: str,
    tabular_severity_path: str,
    batch_size: int = 16,
    image_size: int = 224,
    subset_frac: float = 1.0,
    val_split: float = 0.2,
    num_workers: int = 4,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Create train and validation dataloaders for fusion model."""
    
    # Create individual datasets
    rgb_train_loader, rgb_val_loader, rgb_info = create_rgb_dataloaders(
        metadata_path=rgb_metadata_path,
        images_dir=rgb_images_dir,
        batch_size=1,  # We'll batch in fusion dataset
        image_size=image_size,
        subset_frac=subset_frac,
        balance_classes=True,
        val_split=val_split,
        num_workers=0,
        random_seed=random_seed
    )
    
    thermal_train_loader, thermal_val_loader, thermal_info = create_thermal_dataloaders(
        root_dir=thermal_root_dir,
        batch_size=1,
        image_size=image_size,
        subset_frac=subset_frac,
        val_split=val_split,
        num_workers=0,
        random_seed=random_seed
    )
    
    tabular_train_loader, tabular_val_loader, tabular_info = create_tabular_dataloaders(
        dataset_path=tabular_dataset_path,
        severity_path=tabular_severity_path,
        batch_size=1,
        subset_frac=subset_frac,
        val_split=val_split,
        random_seed=random_seed,
        num_workers=0
    )
    
    # Get the underlying datasets
    rgb_train_ds = rgb_train_loader.dataset
    rgb_val_ds = rgb_val_loader.dataset
    thermal_train_ds = thermal_train_loader.dataset
    thermal_val_ds = thermal_val_loader.dataset
    tabular_train_ds = tabular_train_loader.dataset
    tabular_val_ds = tabular_val_loader.dataset
    
    # Create fusion datasets
    train_dataset = FusionDataset(rgb_train_ds, thermal_train_ds, tabular_train_ds, 'train')
    val_dataset = FusionDataset(rgb_val_ds, thermal_val_ds, tabular_val_ds, 'val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    info = {
        'num_visual_classes': rgb_info['num_classes'],
        'visual_class_names': rgb_info['class_names'],
        'num_triage_classes': 3,
        'tabular_info': tabular_info,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset)
    }
    
    return train_loader, val_loader, info


def train_epoch(model, loader, criteria, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = {'fever': 0.0, 'visual': 0.0, 'triage': 0.0, 'total': 0.0}
    correct = {'fever': 0, 'visual': 0, 'triage': 0}
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        rgb = batch['rgb'].to(device)
        thermal = batch['thermal'].to(device)
        tabular = batch['tabular'].to(device)
        visual_labels = batch['visual_label'].to(device)
        fever_labels = batch['fever_label'].to(device)
        triage_labels = batch['triage_label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(rgb, thermal, tabular)
        
        loss_fever = criteria['fever'](outputs['fever'], fever_labels)
        loss_visual = criteria['visual'](outputs['visual'], visual_labels)
        loss_triage = criteria['triage'](outputs['triage'], triage_labels)
        
        # Weighted loss combination
        loss = loss_visual + 0.5 * loss_fever + 0.5 * loss_triage
        loss.backward()
        optimizer.step()
        
        batch_size = rgb.size(0)
        total += batch_size
        
        running_loss['fever'] += loss_fever.item() * batch_size
        running_loss['visual'] += loss_visual.item() * batch_size
        running_loss['triage'] += loss_triage.item() * batch_size
        running_loss['total'] += loss.item() * batch_size
        
        # Accuracy
        fever_pred = (outputs['fever'] > 0).long()
        _, visual_pred = outputs['visual'].max(1)
        _, triage_pred = outputs['triage'].max(1)
        
        correct['fever'] += fever_pred.eq(fever_labels.long()).sum().item()
        correct['visual'] += visual_pred.eq(visual_labels).sum().item()
        correct['triage'] += triage_pred.eq(triage_labels).sum().item()
        
        pbar.set_postfix({
            'loss': running_loss['total']/total,
            'acc_v': 100.*correct['visual']/total,
            'acc_f': 100.*correct['fever']/total,
            'acc_t': 100.*correct['triage']/total
        })
    
    return {k: v/total for k, v in running_loss.items()}, {k: 100.*v/total for k, v in correct.items()}


def validate(model, loader, criteria, device, epoch):
    """Validate model."""
    model.eval()
    running_loss = {'fever': 0.0, 'visual': 0.0, 'triage': 0.0, 'total': 0.0}
    correct = {'fever': 0, 'visual': 0, 'triage': 0}
    total = 0
    
    all_preds = {'fever': [], 'visual': [], 'triage': []}
    all_labels = {'fever': [], 'visual': [], 'triage': []}
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            rgb = batch['rgb'].to(device)
            thermal = batch['thermal'].to(device)
            tabular = batch['tabular'].to(device)
            visual_labels = batch['visual_label'].to(device)
            fever_labels = batch['fever_label'].to(device)
            triage_labels = batch['triage_label'].to(device)
            
            outputs = model(rgb, thermal, tabular)
            
            loss_fever = criteria['fever'](outputs['fever'], fever_labels)
            loss_visual = criteria['visual'](outputs['visual'], visual_labels)
            loss_triage = criteria['triage'](outputs['triage'], triage_labels)
            
            loss = loss_visual + 0.5 * loss_fever + 0.5 * loss_triage
            
            batch_size = rgb.size(0)
            total += batch_size
            
            running_loss['fever'] += loss_fever.item() * batch_size
            running_loss['visual'] += loss_visual.item() * batch_size
            running_loss['triage'] += loss_triage.item() * batch_size
            running_loss['total'] += loss.item() * batch_size
            
            fever_pred = (outputs['fever'] > 0).long()
            _, visual_pred = outputs['visual'].max(1)
            _, triage_pred = outputs['triage'].max(1)
            
            correct['fever'] += fever_pred.eq(fever_labels.long()).sum().item()
            correct['visual'] += visual_pred.eq(visual_labels).sum().item()
            correct['triage'] += triage_pred.eq(triage_labels).sum().item()
            
            # Store for metrics
            all_preds['fever'].extend(fever_pred.cpu().numpy())
            all_preds['visual'].extend(visual_pred.cpu().numpy())
            all_preds['triage'].extend(triage_pred.cpu().numpy())
            all_labels['fever'].extend(fever_labels.cpu().numpy())
            all_labels['visual'].extend(visual_labels.cpu().numpy())
            all_labels['triage'].extend(triage_labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': running_loss['total']/total,
                'acc_v': 100.*correct['visual']/total,
                'acc_f': 100.*correct['fever']/total,
                'acc_t': 100.*correct['triage']/total
            })
    
    return (
        {k: v/total for k, v in running_loss.items()},
        {k: 100.*v/total for k, v in correct.items()},
        all_preds,
        all_labels
    )


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    set_seed(config.get('seed', 42))
    device = get_device()
    
    # Data
    train_loader, val_loader, info = create_fusion_dataloaders(
        rgb_metadata_path=config['data']['rgb_metadata_path'],
        rgb_images_dir=config['data']['rgb_images_dir'],
        thermal_root_dir=config['data']['thermal_root_dir'],
        tabular_dataset_path=config['data']['tabular_dataset_path'],
        tabular_severity_path=config['data']['tabular_severity_path'],
        batch_size=config['training']['batch_size'],
        image_size=config['data']['image_size'],
        subset_frac=config['data'].get('subset_frac', 1.0),
        val_split=config['data'].get('val_split', 0.2),
        num_workers=config['training'].get('num_workers', 4),
        random_seed=config.get('seed', 42)
    )
    
    # Load pretrained encoders
    rgb_encoder = RGBEncoder(
        pretrained=False,  # We'll load weights
        frozen=config['model']['rgb_frozen'],
        output_dim=config['model']['rgb_output_dim'],
        num_classes=info['num_visual_classes']
    )
    
    thermal_encoder = ThermalEncoder(
        pretrained=False,
        frozen=config['model']['thermal_frozen'],
        output_dim=config['model']['thermal_output_dim'],
        in_channels=1
    )
    
    tabular_encoder = TabularEncoder(
        input_dim=info['tabular_info']['num_symptoms'],
        hidden_dims=config['model']['tabular_hidden_dims'],
        output_dim=config['model']['tabular_output_dim'],
        dropout=config['model'].get('dropout', 0.3)
    )
    
    # Load pretrained weights if available
    if config['model'].get('rgb_checkpoint'):
        ckpt = torch.load(config['model']['rgb_checkpoint'], map_location=device)
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        rgb_encoder.load_state_dict(state, strict=False)
        print(f"Loaded RGB encoder from {config['model']['rgb_checkpoint']}")
    
    if config['model'].get('thermal_checkpoint'):
        ckpt = torch.load(config['model']['thermal_checkpoint'], map_location=device)
        state = ckpt['encoder_state_dict'] if 'encoder_state_dict' in ckpt else ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        thermal_encoder.load_state_dict(state, strict=False)
        print(f"Loaded Thermal encoder from {config['model']['thermal_checkpoint']}")
    
    if config['model'].get('tabular_checkpoint'):
        ckpt = torch.load(config['model']['tabular_checkpoint'], map_location=device)
        # Tabular encoder is part of a larger model
        full_state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        tabular_state = {k.replace('encoder.', ''): v for k, v in full_state.items() if k.startswith('encoder.')}
        # Also try without prefix
        if not tabular_state:
            tabular_state = {k: v for k, v in full_state.items() if not k.startswith('disease_head') and not k.startswith('fever_head') and not k.startswith('triage_head')}
        tabular_encoder.load_state_dict(tabular_state, strict=False)
        print(f"Loaded Tabular encoder from {config['model']['tabular_checkpoint']}")
    
    # Fusion encoder
    fusion_encoder = FusionEncoder(
        rgb_dim=config['model']['rgb_output_dim'],
        thermal_dim=config['model']['thermal_output_dim'],
        tabular_dim=config['model']['tabular_output_dim'],
        fusion_dim=config['model']['fusion_dim'],
        num_heads=config['model'].get('num_heads', 4),
        dropout=config['model'].get('dropout', 0.3)
    )
    
    # Full fusion model
    model = FusionModel(
        rgb_encoder=rgb_encoder,
        thermal_encoder=thermal_encoder,
        tabular_encoder=tabular_encoder,
        fusion_encoder=fusion_encoder,
        num_visual_classes=info['num_visual_classes'],
        num_triage_classes=info['num_triage_classes']
    ).to(device)
    
    # Loss with class weights
    criteria = {
        'visual': nn.CrossEntropyLoss(),
        'fever': nn.BCEWithLogitsLoss(),
        'triage': nn.CrossEntropyLoss()
    }
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['lr']),
        weight_decay=float(config['training'].get('weight_decay', 1e-4))
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs']
    )
    
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(1, config['training']['epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criteria, optimizer, device, epoch)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criteria, device, epoch)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch}: Train Loss={train_loss['total']:.4f}, "
              f"Train Acc: V={train_acc['visual']:.2f}%, F={train_acc['fever']:.2f}%, T={train_acc['triage']:.2f}%")
        print(f"         Val Loss={val_loss['total']:.4f}, "
              f"Val Acc: V={val_acc['visual']:.2f}%, F={val_acc['fever']:.2f}%, T={val_acc['triage']:.2f}%")
        
        # Save best model (based on average of three accuracies)
        avg_val_acc = np.mean([val_acc['visual'], val_acc['fever'], val_acc['triage']])
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config,
                'visual_class_names': info['visual_class_names']
            }, checkpoint_dir / 'best_model.pt')
            print(f"  -> Saved best model (avg_val_acc={avg_val_acc:.2f}%)")
        
        if epoch % config['training'].get('save_every', 5) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
    
    torch.save({
        'epoch': config['training']['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'config': config,
        'visual_class_names': info['visual_class_names']
    }, checkpoint_dir / 'final_model.pt')
    
    np.save(checkpoint_dir / 'history.npy', history)
    
    print(f"\nTraining complete. Best avg val acc: {best_val_acc:.2f}%")
    print(f"Model saved to {checkpoint_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()
    main(args.config)