"""
Evaluate Fusion model and generate evaluation artifacts.
"""
import argparse
import yaml
import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import get_device
from src.data_loaders import (
    create_rgb_dataloaders, create_thermal_dataloaders, create_tabular_dataloaders,
    HAM10000Dataset, SymptomDataset, TerravicDataset, get_rgb_transforms, get_thermal_transforms
)
from src.models import (
    RGBEncoder, ThermalEncoder, TabularEncoder, FusionEncoder, FusionModel
)
from src.evaluation import run_evaluation, plot_training_history


def create_fusion_test_dataset(config, random_seed=42):
    """Create fusion test dataset from individual test splits."""
    # Load individual test datasets
    rgb_full = HAM10000Dataset(
        metadata_path=config['data']['rgb_metadata_path'],
        images_dir=config['data']['rgb_images_dir'],
        transform=None,
        subset_frac=1.0,
        random_seed=random_seed,
        balance_classes=False
    )
    
    thermal_full = TerravicDataset(
        root_dir=config['data']['thermal_root_dir'],
        transform=None,
        subset_frac=1.0,
        random_seed=random_seed
    )
    
    tabular_full = SymptomDataset(
        dataset_path=config['data']['tabular_dataset_path'],
        severity_path=config['data']['tabular_severity_path'],
        fit_encoders=True
    )
    
    # Load or create splits
    def load_or_create_splits(dataset, splits_path, labels, seed):
        if Path(splits_path).exists():
            df = pd.read_csv(splits_path)
            return df[df['split'] == 'test']['index'].values
        indices = np.arange(len(dataset))
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=labels, random_state=seed)
        temp_labels = labels[temp_idx]
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_labels, random_state=seed)
        return test_idx
    
    rgb_splits_path = Path(config['data'].get('rgb_splits_path', 'data/rgb/processed/splits.csv'))
    thermal_splits_path = Path(config['data'].get('thermal_splits_path', 'data/thermal/processed/splits.csv'))
    tabular_splits_path = Path(config['data'].get('tabular_splits_path', 'data/tabular/processed/splits.csv'))
    
    rgb_test_idx = load_or_create_splits(rgb_full, str(rgb_splits_path), rgb_full.df['label'].values, random_seed)
    thermal_test_idx = load_or_create_splits(thermal_full, str(thermal_splits_path), np.zeros(len(thermal_full)), random_seed)
    tabular_test_idx = load_or_create_splits(tabular_full, str(tabular_splits_path), tabular_full.disease_labels, random_seed)
    
    # Use minimum length
    min_len = min(len(rgb_test_idx), len(thermal_test_idx), len(tabular_test_idx))
    
    # Create subset datasets
    rgb_test = Subset(rgb_full, rgb_test_idx[:min_len])
    thermal_test = Subset(thermal_full, thermal_test_idx[:min_len])
    tabular_test = Subset(tabular_full, tabular_test_idx[:min_len])
    
    # Apply transforms
    rgb_test.dataset.transform = get_rgb_transforms(config['data']['image_size'], is_train=False)
    thermal_test.dataset.transform = get_thermal_transforms(config['data']['image_size'], is_train=False)
    
    from src.training.train_fusion import ModalityDropoutDataset
    
    test_dataset = ModalityDropoutDataset(
        rgb_test, thermal_test, tabular_test, 'val',
        modality_dropout_p=0.0, is_train=False
    )
    
    return test_dataset


def main(config_path: str, checkpoint_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = get_device()
    
    # Create test dataset
    test_dataset = create_fusion_test_dataset(config, config.get('seed', 42))
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    # Get info for model construction
    rgb_full = HAM10000Dataset(
        metadata_path=config['data']['rgb_metadata_path'],
        images_dir=config['data']['rgb_images_dir'],
        transform=None,
        subset_frac=1.0,
        random_seed=config.get('seed', 42),
        balance_classes=False
    )
    
    tabular_full = SymptomDataset(
        dataset_path=config['data']['tabular_dataset_path'],
        severity_path=config['data']['tabular_severity_path'],
        fit_encoders=True
    )
    
    info = {
        'num_visual_classes': len(HAM10000Dataset.CLASSES),
        'visual_class_names': HAM10000Dataset.CLASSES,
        'num_triage_classes': 3,
        'tabular_info': {
            'num_symptoms': len(tabular_full.symptom_encoder.classes_),
            'num_diseases': len(tabular_full.disease_encoder.classes_)
        }
    }
    
    # Model
    rgb_encoder = RGBEncoder(
        pretrained=False,
        frozen=False,
        output_dim=config['model']['rgb_output_dim'],
        num_classes=info['num_visual_classes']
    )
    
    thermal_encoder = ThermalEncoder(
        pretrained=False,
        frozen=False,
        output_dim=config['model']['thermal_output_dim'],
        in_channels=1
    )
    
    tabular_encoder = TabularEncoder(
        input_dim=info['tabular_info']['num_symptoms'],
        hidden_dims=config['model']['tabular_hidden_dims'],
        output_dim=config['model']['tabular_output_dim'],
        dropout=config['model'].get('dropout', 0.3)
    )
    
    fusion_encoder = FusionEncoder(
        rgb_dim=config['model']['rgb_output_dim'],
        thermal_dim=config['model']['thermal_output_dim'],
        tabular_dim=config['model']['tabular_output_dim'],
        fusion_dim=config['model']['fusion_dim'],
        num_heads=config['model'].get('num_heads', 4),
        dropout=config['model'].get('dropout', 0.3)
    )
    
    model = FusionModel(
        rgb_encoder=rgb_encoder,
        thermal_encoder=thermal_encoder,
        tabular_encoder=tabular_encoder,
        fusion_encoder=fusion_encoder,
        num_visual_classes=info['num_visual_classes'],
        num_triage_classes=info['num_triage_classes']
    ).to(device)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded model from {checkpoint_path}")
    
    # Evaluate
    results = run_evaluation(
        model=model,
        loader=test_loader,
        device=device,
        save_dir="results/fusion",
        prefix="fusion_",
        is_multitask=True,
        task_names=['fever', 'visual', 'triage'],
        class_names_dict={
            'visual': info['visual_class_names'],
            'fever': ['no_fever', 'fever'],
            'triage': ['mild', 'moderate', 'refer']
        }
    )
    
    # Plot training history if available
    history_path = Path(config['training']['checkpoint_dir']) / 'history.npy'
    if history_path.exists():
        history = np.load(history_path, allow_pickle=True).item()
        plot_training_history(history, "results/fusion/training_curves.png")
    
    print("Fusion evaluation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)