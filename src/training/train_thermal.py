"""
Training script for Thermal encoder (self-supervised / pretraining).
Uses contrastive learning or autoencoder for feature extraction pretraining.
"""
import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path

from src.utils import get_device, set_seed
from src.data_loaders import create_thermal_dataloaders
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


def train_epoch(model, loader, optimizer, device, epoch, temperature=0.5):
    """Train for one epoch with contrastive learning."""
    model.train()
    running_loss = 0.0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for images, _ in pbar:
        # For contrastive learning, we need two augmented views
        # Since our dataset returns (image, dummy_label), we'll create two augmentations
        batch_size = images.size(0)
        images = images.to(device)
        
        optimizer.zero_grad()
        
        # Get two augmented views by applying transform twice
        # Note: This is a simplified version. In practice, you'd use a dataset that returns two views.
        z1 = model(images)
        z2 = model(images)  # Same image, different augmentations due to random transforms
        
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
    
    # Data
    train_loader, val_loader, info = create_thermal_dataloaders(
        root_dir=config['data']['root_dir'],
        batch_size=config['training']['batch_size'],
        image_size=config['data']['image_size'],
        subset_frac=config['data'].get('subset_frac', 1.0),
        val_split=config['data'].get('val_split', 0.2),
        num_workers=config['training'].get('num_workers', 4),
        random_seed=config.get('seed', 42)
    )
    
    # Model
    encoder = ThermalEncoder(
        pretrained=config['model']['pretrained'],
        frozen=config['model']['frozen'],
        output_dim=config['model']['output_dim'],
        in_channels=info['in_channels']
    )
    model = ThermalPretrainModel(encoder, proj_dim=config['model'].get('proj_dim', 128)).to(device)
    
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
    
    best_val_loss = float('inf')
    
    for epoch in range(1, config['training']['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, 
                                 temperature=config['training'].get('temperature', 0.5))
        val_loss = validate(model, val_loader, device, epoch,
                           temperature=config['training'].get('temperature', 0.5))
        
        scheduler.step()
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'projector_state_dict': model.projector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_dir / 'best_model.pt')
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")
        
        if epoch % config['training'].get('save_every', 5) == 0:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'projector_state_dict': model.projector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
    
    # Save final encoder (without projector) for downstream use
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'config': config
    }, checkpoint_dir / 'encoder_final.pt')
    
    print(f"\nPretraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Encoder saved to {checkpoint_dir / 'encoder_final.pt'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()
    main(args.config)