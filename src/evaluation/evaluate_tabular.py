"""
Evaluate Tabular baseline model and generate evaluation artifacts.
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
from src.data_loaders import SymptomDataset
from src.models import TabularEncoder
from src.evaluation import run_evaluation, plot_training_history


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


def create_test_split(dataset: SymptomDataset, splits_path: str, seed: int = 42):
    """Create or load test split for tabular data."""
    if Path(splits_path).exists():
        splits_df = pd.read_csv(splits_path)
        test_idx = splits_df[splits_df['split'] == 'test']['index'].values
        return test_idx
    
    # Create splits if not exist
    disease_labels = dataset.disease_labels
    indices = np.arange(len(dataset))
    
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=disease_labels, random_state=seed)
    temp_labels = disease_labels[temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_labels, random_state=seed)
    
    return test_idx


def main(config_path: str, checkpoint_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = get_device()
    
    # Create full dataset
    full_dataset = SymptomDataset(
        dataset_path=config['data']['dataset_path'],
        severity_path=config['data']['severity_path'],
        fit_encoders=True
    )
    
    # Create or load test split
    splits_path = Path(config['data'].get('splits_path', 'data/tabular/processed/splits.csv'))
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    
    test_idx = create_test_split(full_dataset, str(splits_path), config.get('seed', 42))
    test_dataset = Subset(full_dataset, test_idx)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=True
    )
    
    # Model
    encoder = TabularEncoder(
        input_dim=full_dataset.symptom_encoder.classes_.shape[0],
        hidden_dims=config['model']['hidden_dims'],
        output_dim=config['model']['output_dim'],
        dropout=config['model'].get('dropout', 0.3)
    )
    model = TabularClassifier(encoder, len(full_dataset.disease_encoder.classes_)).to(device)
    
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
    
    info = {
        'disease_names': disease_encoder.classes_.tolist(),
        'num_diseases': len(disease_encoder.classes_),
        'num_symptoms': len(symptom_encoder.classes_)
    }
    
    # Evaluate
    results = run_evaluation(
        model=model,
        loader=test_loader,
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
    
    # Plot training history if available
    history_path = Path(config['training']['checkpoint_dir']) / 'history.npy'
    if history_path.exists():
        history = np.load(history_path, allow_pickle=True).item()
        plot_training_history(history, "results/tabular/training_curves.png")
    
    print("Tabular evaluation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)