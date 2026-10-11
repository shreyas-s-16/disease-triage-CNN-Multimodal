"""
Evaluate RGB baseline model.
"""
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.utils import get_device
from src.data_loaders import create_rgb_dataloaders
from src.models import RGBEncoder
from src.evaluation import run_evaluation


def main(config_path: str, checkpoint_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = get_device()
    
    # Data (use full validation set, no subset)
    _, val_loader, info = create_rgb_dataloaders(
        metadata_path=config['data']['metadata_path'].replace('balanced_subset', ''),
        images_dir=config['data']['images_dir'],
        batch_size=config['training']['batch_size'],
        image_size=config['data']['image_size'],
        subset_frac=1.0,
        balance_classes=False,
        val_split=config['data'].get('val_split', 0.2),
        num_workers=config['training'].get('num_workers', 4),
        random_seed=config.get('seed', 42)
    )
    
    # Model
    model = RGBEncoder(
        pretrained=False,
        frozen=False,
        output_dim=config['model']['output_dim'],
        num_classes=info['num_classes']
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
        save_dir="results/rgb",
        prefix="rgb_",
        class_names=info['class_names']
    )
    
    print("RGB evaluation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)