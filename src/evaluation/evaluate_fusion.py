"""
Evaluate Fusion model.
"""
import argparse
import yaml
import torch
import pickle
from torch.utils.data import DataLoader
from pathlib import Path

from src.utils import get_device
from src.data_loaders import (
    create_rgb_dataloaders, create_thermal_dataloaders, create_tabular_dataloaders
)
from src.models import (
    RGBEncoder, ThermalEncoder, TabularEncoder, FusionEncoder, FusionModel
)
from src.evaluation import run_evaluation
from src.training.train_fusion import FusionDataset


def main(config_path: str, checkpoint_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = get_device()
    
    # Create individual validation dataloaders
    _, rgb_val_loader, rgb_info = create_rgb_dataloaders(
        metadata_path=config['data']['rgb_metadata_path'].replace('balanced_subset', ''),
        images_dir=config['data']['rgb_images_dir'],
        batch_size=1,
        image_size=config['data']['image_size'],
        subset_frac=1.0,
        balance_classes=False,
        val_split=config['data'].get('val_split', 0.2),
        num_workers=0,
        random_seed=config.get('seed', 42)
    )
    
    _, thermal_val_loader, thermal_info = create_thermal_dataloaders(
        root_dir=config['data']['thermal_root_dir'],
        batch_size=1,
        image_size=config['data']['image_size'],
        subset_frac=1.0,
        val_split=config['data'].get('val_split', 0.2),
        num_workers=0,
        random_seed=config.get('seed', 42)
    )
    
    _, tabular_val_loader, tabular_info = create_tabular_dataloaders(
        dataset_path=config['data']['tabular_dataset_path'],
        severity_path=config['data']['tabular_severity_path'],
        batch_size=1,
        subset_frac=1.0,
        val_split=config['data'].get('val_split', 0.2),
        random_seed=config.get('seed', 42),
        num_workers=0
    )
    
    # Create fusion validation dataset
    rgb_val_ds = rgb_val_loader.dataset
    thermal_val_ds = thermal_val_loader.dataset
    tabular_val_ds = tabular_val_loader.dataset
    
    val_dataset = FusionDataset(rgb_val_ds, thermal_val_ds, tabular_val_ds, 'val')
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    # Model
    rgb_encoder = RGBEncoder(
        pretrained=False,
        frozen=False,
        output_dim=config['model']['rgb_output_dim'],
        num_classes=rgb_info['num_classes']
    )
    
    thermal_encoder = ThermalEncoder(
        pretrained=False,
        frozen=False,
        output_dim=config['model']['thermal_output_dim'],
        in_channels=1
    )
    
    tabular_encoder = TabularEncoder(
        input_dim=tabular_info['num_symptoms'],
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
        num_visual_classes=rgb_info['num_classes'],
        num_triage_classes=3
    ).to(device)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    print(f"Loaded model from {checkpoint_path}")
    
    # Evaluate
    results = run_evaluation(
        model=model,
        loader=val_loader,
        device=device,
        save_dir="results/fusion",
        prefix="fusion_",
        is_multitask=True,
        task_names=['fever', 'visual', 'triage'],
        class_names_dict={
            'visual': rgb_info['class_names'],
            'fever': ['no_fever', 'fever'],
            'triage': ['mild', 'moderate', 'refer']
        }
    )
    
    print("Fusion evaluation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)