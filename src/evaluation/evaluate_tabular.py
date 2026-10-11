"""
Evaluate Tabular baseline model.
"""
import argparse
import yaml
import torch
import pickle
from torch.utils.data import DataLoader
from pathlib import Path

from src.utils import get_device
from src.data_loaders import create_tabular_dataloaders, SymptomDataset
from src.models import TabularEncoder
from src.evaluation import run_evaluation


class TabularClassifier(torch.nn.Module):
    """Tabular encoder + classification heads."""
    
    def __init__(self, encoder: TabularEncoder, num_diseases: int, num_triage: int = 3):
        super().__init__()
        self.encoder = encoder
        self.disease_head = torch.nn.Linear(encoder.output_dim, num_diseases)
        self.fever_head = torch.nn.Linear(encoder.output_dim, 1)
        self.triage_head = torch.nn.Linear(encoder.output_dim, num_triage)
    
    def forward(self, x: torch.Tensor) -> dict:
        features = self.encoder(x)
        return {
            'disease': self.disease_head(features),
            'fever': self.fever_head(features).squeeze(-1),
            'triage': self.triage_head(features)
        }


def main(config_path: str, checkpoint_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = get_device()
    
    # Data
    _, val_loader, info = create_tabular_dataloaders(
        dataset_path=config['data']['dataset_path'],
        severity_path=config['data']['severity_path'],
        batch_size=config['training']['batch_size'],
        subset_frac=1.0,
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
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    print(f"Loaded model from {checkpoint_path}")
    
    # Load encoders for class names
    with open(Path(checkpoint_path).parent / 'disease_encoder.pkl', 'rb') as f:
        disease_encoder = pickle.load(f)
    with open(Path(checkpoint_path).parent / 'symptom_encoder.pkl', 'rb') as f:
        symptom_encoder = pickle.load(f)
    
    # Evaluate
    results = run_evaluation(
        model=model,
        loader=val_loader,
        device=device,
        save_dir="results/tabular",
        prefix="tabular_",
        is_multitask=True,
        task_names=['disease', 'fever', 'triage'],
        class_names_dict={
            'disease': info['disease_names'],
            'fever': ['no_fever', 'fever'],
            'triage': ['mild', 'moderate', 'refer']
        }
    )
    
    print("Tabular evaluation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)