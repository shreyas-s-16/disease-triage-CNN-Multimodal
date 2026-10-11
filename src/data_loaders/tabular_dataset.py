"""
Tabular (Disease Symptom Prediction) Dataset Loader
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from typing import Optional, Tuple, List, Dict
import pickle


class SymptomDataset(Dataset):
    """Disease Symptom Prediction dataset."""
    
    def __init__(
        self,
        dataset_path: str,
        severity_path: str,
        symptom_encoder: Optional[MultiLabelBinarizer] = None,
        disease_encoder: Optional[LabelEncoder] = None,
        fit_encoders: bool = True,
        fever_threshold: str = 'high_fever'
    ):
        """
        Args:
            dataset_path: Path to dataset.csv (Disease + 17 symptom columns)
            severity_path: Path to Symptom-severity.csv
            symptom_encoder: Pre-fitted MultiLabelBinarizer for symptoms
            disease_encoder: Pre-fitted LabelEncoder for diseases
            fit_encoders: Whether to fit encoders on this data
            fever_threshold: Symptom to use as fever proxy ('high_fever', 'mild_fever', 'chills', 'shivering')
        """
        self.df = pd.read_csv(dataset_path)
        self.severity_df = pd.read_csv(severity_path)
        self.fever_threshold = fever_threshold
        
        # Symptom columns (all except 'Disease')
        self.symptom_cols = [c for c in self.df.columns if c != 'Disease']
        
        # Convert symptoms to list format
        self.symptoms_list = []
        for _, row in self.df.iterrows():
            symptoms = row[self.symptom_cols].dropna().astype(str).str.strip().tolist()
            self.symptoms_list.append(symptoms)
        
        # Encode diseases
        self.diseases = self.df['Disease'].astype(str).str.strip().values
        if fit_encoders:
            self.disease_encoder = LabelEncoder()
            self.disease_labels = self.disease_encoder.fit_transform(self.diseases)
            self.symptom_encoder = MultiLabelBinarizer()
            self.symptom_encoder.fit(self.symptoms_list)
        else:
            self.disease_encoder = disease_encoder
            self.symptom_encoder = symptom_encoder
            self.disease_labels = self.disease_encoder.transform(self.diseases)
        
        # Create symptom feature matrix (multi-hot)
        self.symptom_features = self.symptom_encoder.transform(self.symptoms_list).astype(np.float32)
        
        # Create fever proxy label (binary: has high_fever symptom)
        self.fever_labels = np.array([
            1 if fever_threshold in symptoms else 0 
            for symptoms in self.symptoms_list
        ], dtype=np.float32)
        
        # Create triage labels based on fever + disease severity
        # Simple heuristic: high_fever -> refer, mild_fever/chills -> moderate, else mild
        self.triage_labels = np.zeros(len(self.df), dtype=np.int64)
        for i, symptoms in enumerate(self.symptoms_list):
            if 'high_fever' in symptoms:
                self.triage_labels[i] = 2  # refer
            elif 'mild_fever' in symptoms or 'chills' in symptoms or 'shivering' in symptoms:
                self.triage_labels[i] = 1  # moderate
            else:
                self.triage_labels[i] = 0  # mild
        
        print(f"Tabular dataset: {len(self.df)} samples")
        print(f"Number of diseases: {len(self.disease_encoder.classes_)}")
        print(f"Number of unique symptoms: {len(self.symptom_encoder.classes_)}")
        print(f"Fever proxy ({fever_threshold}) positive: {self.fever_labels.sum()}/{len(self.fever_labels)}")
        print(f"Triage distribution: mild={np.sum(self.triage_labels==0)}, moderate={np.sum(self.triage_labels==1)}, refer={np.sum(self.triage_labels==2)}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'symptoms': torch.tensor(self.symptom_features[idx], dtype=torch.float32),
            'disease': torch.tensor(self.disease_labels[idx], dtype=torch.long),
            'fever': torch.tensor(self.fever_labels[idx], dtype=torch.float32),
            'triage': torch.tensor(self.triage_labels[idx], dtype=torch.long)
        }
    
    def get_disease_weights(self) -> torch.Tensor:
        """Compute class weights for disease classification."""
        class_counts = np.bincount(self.disease_labels)
        total = len(self.disease_labels)
        weights = total / (len(class_counts) * class_counts)
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_fever_weights(self) -> torch.Tensor:
        """Compute class weights for fever binary classification."""
        pos = self.fever_labels.sum()
        neg = len(self.fever_labels) - pos
        total = len(self.fever_labels)
        weights = torch.tensor([total / (2 * neg), total / (2 * pos)], dtype=torch.float32)
        return weights
    
    def get_triage_weights(self) -> torch.Tensor:
        """Compute class weights for triage classification."""
        class_counts = np.bincount(self.triage_labels, minlength=3)
        total = len(self.triage_labels)
        weights = total / (3 * class_counts)
        return torch.tensor(weights, dtype=torch.float32)
    
    def save_encoders(self, path: str):
        """Save encoders for inference."""
        with open(path, 'wb') as f:
            pickle.dump({
                'disease_encoder': self.disease_encoder,
                'symptom_encoder': self.symptom_encoder
            }, f)
    
    @classmethod
    def load_encoders(cls, path: str) -> Tuple[LabelEncoder, MultiLabelBinarizer]:
        """Load encoders for inference."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data['disease_encoder'], data['symptom_encoder']


def create_tabular_dataloaders(
    dataset_path: str,
    severity_path: str,
    batch_size: int = 64,
    subset_frac: float = 1.0,
    val_split: float = 0.2,
    random_seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Create train and validation dataloaders for tabular data."""
    
    # Full dataset
    full_dataset = SymptomDataset(
        dataset_path=dataset_path,
        severity_path=severity_path,
        fit_encoders=True
    )
    
    # Split indices
    np.random.seed(random_seed)
    indices = np.random.permutation(len(full_dataset))
    val_size = int(len(full_dataset) * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
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
    
    return train_loader, val_loader, info