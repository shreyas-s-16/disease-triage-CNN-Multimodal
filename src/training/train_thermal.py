"""
Training script for Thermal encoder (self-supervised / pretraining).
Uses contrastive learning for feature extraction pretraining.
Supports early stopping and resumable checkpoints.
"""
import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.utils import get_device, set_seed
from src.data_loaders import TerravicDataset, get_thermal_transforms
from src.models import ThermalEncoder


class ThermalPretrainModel(nn.Module):
    """Thermal encoder with projection head for contrastive learning."""
    
    def __init__(self, encoder: ThermalEncoder, proj_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(encoder.output_dim, encoder.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(encoder.output_dim, proj_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.projector(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Normalized Temperature-scaled Cross Entropy Loss (SimCLR)."""
    batch_size = z1.size(0)
    
    # Normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Concatenate
    representations = torch.cat([z1, z2], dim=0)  # [2B, D]
    
    # Similarity matrix
    similarity = torch.mm(representations, representations.t()) / temperature  # [2B, 2B]
    
    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
    similarity.masked_fill_(mask, -float('inf'))
    
    # Labels: for each sample in first batch, positive is corresponding in second batch
    labels = torch.arange(batch_size, device=z1.device)
    labels = torch.cat([labels + batch_size, labels])  # [2B]
    
    loss = F.cross_entropy(similarity, labels)
    return loss


def create_splits(dataset, splits_path: str, seed: int = 42):
    """Create train/val/test splits for thermal data."""
    indices = np.arange(len(dataset))
    
    # First split: 70% train, 30% temp
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=seed)
    
    # Second split: 15% val, 15% test from temp
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed)
    
    # Save splits
    import pandas as pd
    splits_df = pd.DataFrame({
        'split': ['train'] * len(train_idx) + ['val'] * len(val_idx) + ['test'] * len(test_idx),
        'index': list(train_idx) + list(val_idx) + list(test_idx)
    })
    splits_df.to_csv(splits_path, index=False)
    print(f"Saved thermal splits to {splits_path}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    return train_idx, val_idx, test_idx


def load_splits(splits_path: str):
    """Load existing splits from CSV."""
    import pandas as pd
    splits_df = pd.read_csv(splits_path)
    train_idx = splits_df[splits_df['split'] == 'train']['index'].values
    val_idx = splits_df[splits_df['split'] == 'val']['index'].values
    test_idx = splits_df[splits_df['split'] == 'test']['index'].values
    return train_idx, val_idx, test_idx


def train_epoch(model, loader, optimizer, device, epoch, temperature=0.5):
    """Train for one epoch with contrastive learning."""
    model.train()
    running_loss = 0.0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for images, _ in pbar:
        batch_size = images.size(0)
        images = images.to(device)
        
        optimizer.zero_grad()
        
        z1 = model(images)
        z2 = model(images)
        
        loss = nt_xent_loss(z1, z2, temperature)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_size
        total += batch_size
        
        pbar.set_postfix({'loss': running_loss/total})
    
    return running_loss / total


def validate(model, loader, device, epoch, temperature=0.5):
    """Validate with contrastive loss."""
    model.eval()
    running_loss = 0.0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
        for images, _ in pbar:
            images = images.to(device)
            z1 = model(images)
            z2 = model(images)
            loss = nt_xent_loss(z1, z2, temperature)
            
            running_loss += loss.item() * images.size(0)
            total += images.size(0)
            
            pbar.set_postfix({'loss': running_loss/total})
    
    return running_loss / total


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    set_seed(config.get('seed', 42))
    device = get_device()
    
    # Create full dataset
    full_dataset = TerravicDataset(
        root_dir=config['data']['root_dir'],
        transform=None,
        subset_frac=config['data'].get('subset_frac', 1.0),
        random_seed=config.get('seed', 42)
    )
    
    # Create or load splits
    splits_path = Path(config['data'].get('splits_path', 'data/thermal/processed/splits.csv'))
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    
    if splits_path.exists():
        train_idx, val_idx, test_idx = load_splits(str(splits_path))
        print(f"Loaded existing thermal splits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    else:
        train_idx, val_idx, test_idx = create_splits(full_dataset, str(splits_path), config.get('seed', 42))
    
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    
    train_transform = get_thermal_transforms(config['data']['image_size'], is_train=True)
    val_transform = get_thermal_transforms(config['data']['image_size'], is_train=False)
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
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
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'image_size': config['data']['image_size'],
        'in_channels': 1
    }
    
    # Model
    encoder = ThermalEncoder(
        pretrained=config['model']['pretrained'],
        frozen=config['model']['frozen'],
        output_dim=config['model']['output_dim'],
        in_channels=info['in_channels']
    )
    model = ThermalPretrainModel(encoder, proj_dim=config['model'].get('proj_dim', 128)).to(device)
    
    # Resume from checkpoint if exists
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_path = checkpoint_dir / 'checkpoint_last.pt'
    start_epoch = 1
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    if resume_path.exists():
        print(f"Resuming from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)
        history = checkpoint.get('history', history)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['lr']),
        weight_decay=float(config['training'].get('weight_decay', 1e-4))
    )
    
    if resume_path.exists() and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs']
    )
    
    if resume_path.exists() and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        for _ in range(start_epoch - 1):
            scheduler.step()
    
    # Early stopping config
    early_stopping_patience = config['training'].get('early_stopping_patience', 5)
    early_stopping_min_delta = config['training'].get('early_stopping_min_delta', 1e-4)
    
    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, 
                                 temperature=config['training'].get('temperature', 0.5))
        val_loss = validate(model, val_loader, device, epoch,
                           temperature=config['training'].get('temperature', 0.5))
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        if val_loss < best_val_loss - config['training'].get('early_stopping_min_delta', 1e-4):
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'projector_state_dict': model.projector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'config': config,
                'history': history
            }, checkpoint_dir / 'best_model.pt')
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['training'].get('early_stopping_patience', 5):
                print(f"Early stopping triggered after {epoch} epochs (patience={config['training'].get('early_stopping_patience', 5)})")
                break
        
        # Save last checkpoint (for resuming)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'encoder_state_dict': encoder.state_dict(),
            'projector_state_dict': model.projector.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'config': config,
            'history': history
        }, checkpoint_dir / 'checkpoint_last.pt')
    
    # Save final encoder (without projector) for downstream use
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'config': config
    }, checkpoint_dir / 'encoder_final.pt')
    
    # Save history
    np.save(checkpoint_dir / 'history.npy', history)
    
    print(f"\nPretraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Encoder saved to {checkpoint_dir / 'encoder_final.pt'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()
    main(args.config)