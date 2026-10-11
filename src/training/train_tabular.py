"""
Training script for Tabular baseline model.
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
import pickle

from src.utils import get_device, set_seed
from src.data_loaders import create_tabular_dataloaders
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
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    set_seed(config.get('seed', 42))
    device = get_device()
    
    # Data
    train_loader, val_loader, info = create_tabular_dataloaders(
        dataset_path=config['data']['dataset_path'],
        severity_path=config['data']['severity_path'],
        batch_size=config['training']['batch_size'],
        subset_frac=config['data'].get('subset_frac', 1.0),
        val_split=config['data'].get('val_split', 0.2),
        random_seed=config.get('seed', 42),
        num_workers=config['training'].get('num_workers', 0)
    )
    
    # Model
    encoder = TabularEncoder(
        input_dim=info['num_symptoms'],
        hidden_dims=config['model']['hidden_dims'],
        output_dim=config['model']['output_dim'],
        dropout=config['model'].get('dropout', 0.3)
    )
    model = TabularClassifier(encoder, info['num_diseases']).to(device)
    
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
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs']
    )
    
    # Checkpoint dir
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save encoders for inference
    with open(checkpoint_dir / 'disease_encoder.pkl', 'wb') as f:
        pickle.dump(info['disease_encoder'], f)
    with open(checkpoint_dir / 'symptom_encoder.pkl', 'wb') as f:
        pickle.dump(info['symptom_encoder'], f)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(1, config['training']['epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criteria, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criteria, device, epoch)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch}: Train Loss={train_loss}, Train Acc={train_acc}")
        print(f"         Val Loss={val_loss}, Val Acc={val_acc}")
        
        # Save best model (based on disease accuracy)
        if val_acc['disease'] > best_val_acc:
            best_val_acc = val_acc['disease']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config,
                'disease_names': info['disease_names']
            }, checkpoint_dir / 'best_model.pt')
            print(f"  -> Saved best model (val_disease_acc={val_acc['disease']:.2f}%)")
        
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
        'disease_names': info['disease_names']
    }, checkpoint_dir / 'final_model.pt')
    
    np.save(checkpoint_dir / 'history.npy', history)
    
    print(f"\nTraining complete. Best val disease acc: {best_val_acc:.2f}%")
    print(f"Model saved to {checkpoint_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()
    main(args.config)