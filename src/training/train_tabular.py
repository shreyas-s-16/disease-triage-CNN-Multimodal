"""
Training script for Tabular baseline model.
Supports early stopping, resumable checkpoints, and proper loss weighting.
"""
import os
import argparse
import yaml
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split

from src.utils import get_device, set_seed
from src.data_loaders import SymptomDataset
from src.models import TabularEncoder


class TabularClassifier(nn.Module):
    """Tabular encoder + classification heads."""
    
    def __init__(self, encoder: TabularEncoder, num_diseases: int, num_triage: int = 3):
        super().__init__()
        self.encoder = encoder
        self.disease_head = nn.Linear(encoder.output_dim, num_diseases)
        self.fever_head = nn.Linear(encoder.output_dim, 1)
        self.triage_head = nn.Linear(encoder.output_dim, num_triage)
    
    def forward(self, x: torch.Tensor) -> dict:
        features = self.encoder(x)
        return {
            'disease': self.disease_head(features),
            'fever': self.fever_head(features).squeeze(-1),
            'triage': self.triage_head(features)
        }


def create_stratified_splits(dataset: SymptomDataset, splits_path: str, seed: int = 42):
    """Create stratified train/val/test splits for tabular data."""
    disease_labels = dataset.disease_labels
    indices = np.arange(len(dataset))
    
    # First split: 70% train, 30% temp
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, stratify=disease_labels, random_state=seed
    )
    
    # Second split: 15% val, 15% test from temp
    temp_labels = disease_labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=seed
    )
    
    # Save splits
    splits_df = pd.DataFrame({
        'split': ['train'] * len(train_idx) + ['val'] * len(val_idx) + ['test'] * len(test_idx),
        'index': list(train_idx) + list(val_idx) + list(test_idx)
    })
    splits_df.to_csv(splits_path, index=False)
    print(f"Saved tabular splits to {splits_path}")
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


def train_epoch(model, loader, criteria, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = {'disease': 0.0, 'fever': 0.0, 'triage': 0.0}
    correct = {'disease': 0, 'fever': 0, 'triage': 0}
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        symptoms = batch['symptoms'].to(device)
        disease_labels = batch['disease'].to(device)
        fever_labels = batch['fever'].to(device)
        triage_labels = batch['triage'].to(device)
        
        optimizer.zero_grad()
        outputs = model(symptoms)
        
        loss_disease = criteria['disease'](outputs['disease'], disease_labels)
        loss_fever = criteria['fever'](outputs['fever'], fever_labels)
        loss_triage = criteria['triage'](outputs['triage'], triage_labels)
        
        # Weighted sum of losses
        loss = loss_disease + 0.5 * loss_fever + 0.5 * loss_triage
        loss.backward()
        optimizer.step()
        
        batch_size = symptoms.size(0)
        total += batch_size
        
        running_loss['disease'] += loss_disease.item() * batch_size
        running_loss['fever'] += loss_fever.item() * batch_size
        running_loss['triage'] += loss_triage.item() * batch_size
        
        _, pred_disease = outputs['disease'].max(1)
        pred_fever = (outputs['fever'] > 0).long().squeeze()
        _, pred_triage = outputs['triage'].max(1)
        
        correct['disease'] += pred_disease.eq(disease_labels).sum().item()
        correct['fever'] += pred_fever.eq(fever_labels.long()).sum().item()
        correct['triage'] += pred_triage.eq(triage_labels).sum().item()
        
        pbar.set_postfix({
            'loss_d': running_loss['disease']/total,
            'loss_f': running_loss['fever']/total,
            'loss_t': running_loss['triage']/total,
            'acc_d': 100.*correct['disease']/total,
            'acc_f': 100.*correct['fever']/total,
            'acc_t': 100.*correct['triage']/total
        })
    
    return {k: v/total for k, v in running_loss.items()}, {k: 100.*v/total for k, v in correct.items()}


def validate(model, loader, criteria, device, epoch):
    """Validate model."""
    model.eval()
    running_loss = {'disease': 0.0, 'fever': 0.0, 'triage': 0.0}
    correct = {'disease': 0, 'fever': 0, 'triage': 0}
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            symptoms = batch['symptoms'].to(device)
            disease_labels = batch['disease'].to(device)
            fever_labels = batch['fever'].to(device)
            triage_labels = batch['triage'].to(device)
            
            outputs = model(symptoms)
            
            loss_disease = criteria['disease'](outputs['disease'], disease_labels)
            loss_fever = criteria['fever'](outputs['fever'], fever_labels)
            loss_triage = criteria['triage'](outputs['triage'], triage_labels)
            
            batch_size = symptoms.size(0)
            total += batch_size
            
            running_loss['disease'] += loss_disease.item() * batch_size
            running_loss['fever'] += loss_fever.item() * batch_size
            running_loss['triage'] += loss_triage.item() * batch_size
            
            _, pred_disease = outputs['disease'].max(1)
            pred_fever = (outputs['fever'] > 0).long().squeeze()
            _, pred_triage = outputs['triage'].max(1)
            
            correct['disease'] += pred_disease.eq(disease_labels).sum().item()
            correct['fever'] += pred_fever.eq(fever_labels.long()).sum().item()
            correct['triage'] += pred_triage.eq(triage_labels).sum().item()
            
            pbar.set_postfix({
                'loss_d': running_loss['disease']/total,
                'acc_d': 100.*correct['disease']/total,
                'acc_f': 100.*correct['fever']/total,
                'acc_t': 100.*correct['triage']/total
            })
    
    return {k: v/total for k, v in running_loss.items()}, {k: 100.*v/total for k, v in correct.items()}


def main(config_path: str):
    import pandas as pd
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    set_seed(config.get('seed', 42))
    device = get_device()
    
    # Create full dataset
    full_dataset = SymptomDataset(
        dataset_path=config['data']['dataset_path'],
        severity_path=config['data']['severity_path'],
        fit_encoders=True
    )
    
    # Create or load stratified splits
    splits_path = Path(config['data'].get('splits_path', 'data/tabular/processed/splits.csv'))
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    
    if splits_path.exists():
        train_idx, val_idx, test_idx = load_splits(str(splits_path))
        print(f"Loaded existing tabular splits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    else:
        train_idx, val_idx, test_idx = create_stratified_splits(full_dataset, str(splits_path), config.get('seed', 42))
    
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=True
    )
    
    info = {
        'num_diseases': len(full_dataset.disease_encoder.classes_),
        'num_symptoms': len(full_dataset.symptom_encoder.classes_),
        'symptom_names': full_dataset.symptom_encoder.classes_.tolist(),
        'disease_names': full_dataset.disease_encoder.classes_.tolist(),
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'disease_weights': full_dataset.get_disease_weights(),
        'fever_weights': full_dataset.get_fever_weights(),
        'triage_weights': full_dataset.get_triage_weights(),
        'disease_encoder': full_dataset.disease_encoder,
        'symptom_encoder': full_dataset.symptom_encoder
    }
    
    # Model
    encoder = TabularEncoder(
        input_dim=info['num_symptoms'],
        hidden_dims=config['model']['hidden_dims'],
        output_dim=config['model']['output_dim'],
        dropout=config['model'].get('dropout', 0.3)
    )
    model = TabularClassifier(encoder, info['num_diseases']).to(device)
    
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
    criteria = {
        'disease': nn.CrossEntropyLoss(weight=info['disease_weights'].to(device)),
        'fever': nn.BCEWithLogitsLoss(pos_weight=info['fever_weights'][1].to(device)),
        'triage': nn.CrossEntropyLoss(weight=info['triage_weights'].to(device))
    }
    
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
    
    # Save encoders for inference
    with open(checkpoint_dir / 'disease_encoder.pkl', 'wb') as f:
        pickle.dump(info['disease_encoder'], f)
    with open(checkpoint_dir / 'symptom_encoder.pkl', 'wb') as f:
        pickle.dump(info['symptom_encoder'], f)
    
    # Early stopping config
    early_stopping_patience = config['training'].get('early_stopping_patience', 5)
    early_stopping_min_delta = config['training'].get('early_stopping_min_delta', 1e-4)
    
    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criteria, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criteria, device, epoch)
        
        scheduler.step()
        
        # Use combined loss for early stopping
        train_total_loss = sum(train_loss.values())
        val_total_loss = sum(val_loss.values())
        
        history['train_loss'].append(train_total_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_total_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch}: Train Loss={train_total_loss:.4f}, Train Acc={train_acc}")
        print(f"         Val Loss={val_total_loss:.4f}, Val Acc={val_acc}")
        
        # Save best model (based on disease accuracy)
        if val_acc['disease'] > best_val_acc:
            best_val_acc = val_acc['disease']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_total_loss,
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss,
                'config': config,
                'disease_names': info['disease_names'],
                'history': history
            }, checkpoint_dir / 'best_model.pt')
            print(f"  -> Saved best model (val_disease_acc={val_acc['disease']:.2f}%)")
        
        # Save last checkpoint (for resuming)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_total_loss,
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'config': config,
            'history': history
        }, checkpoint_dir / 'checkpoint_last.pt')
        
        # Early stopping check
        if val_total_loss < best_val_loss - config['training'].get('early_stopping_min_delta', 1e-4):
            best_val_loss = val_total_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['training'].get('early_stopping_patience', 5):
                print(f"Early stopping triggered after {epoch} epochs (patience={config['training'].get('early_stopping_patience', 5)})")
                break
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_total_loss,
        'best_val_acc': best_val_acc,
        'config': config,
        'disease_names': info['disease_names'],
        'history': history
    }, checkpoint_dir / 'final_model.pt')
    
    np.save(checkpoint_dir / 'history.npy', history)
    
    print(f"\nTraining complete. Best val disease acc: {best_val_acc:.2f}%")
    print(f"Model saved to {checkpoint_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()
    main(args.config)