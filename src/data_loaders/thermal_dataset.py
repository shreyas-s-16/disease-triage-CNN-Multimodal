"""
Thermal (Terravic Facial IR) Dataset Loader
Used for pretraining thermal CNN encoder only (no fever labels available)
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Optional, Tuple, List
import numpy as np
from pathlib import Path


class TerravicDataset(Dataset):
    """Terravic Facial IR thermal face dataset for pretraining."""
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        subset_frac: float = 1.0,
        random_seed: int = 42
    ):
        """
        Args:
            root_dir: Root directory containing extracted faceXX folders
            transform: Image transforms
            subset_frac: Fraction of data to use
            random_seed: Random seed
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Find all JPEG images
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']:
            self.image_paths.extend(list(self.root_dir.rglob(ext)))
        
        self.image_paths = [str(p) for p in self.image_paths]
        
        if subset_frac < 1.0:
            np.random.seed(random_seed)
            n_samples = max(1, int(len(self.image_paths) * subset_frac))
            self.image_paths = np.random.choice(self.image_paths, n_samples, replace=False).tolist()
        
        print(f"Terravic dataset: {len(self.image_paths)} thermal images")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.image_paths[idx]
        
        # Load as grayscale (thermal images are single channel)
        image = Image.open(image_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        # Return image and dummy label (for self-supervised or encoder pretraining)
        return image, torch.tensor(0, dtype=torch.long)


def get_thermal_transforms(image_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """Get transforms for thermal images (single channel)."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            # Normalize for single channel (ImageNet stats adapted)
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])


def create_thermal_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    subset_frac: float = 1.0,
    val_split: float = 0.2,
    num_workers: int = 4,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, dict]:
    """Create train and validation dataloaders for thermal data."""
    
    full_dataset = TerravicDataset(
        root_dir=root_dir,
        transform=None,
        subset_frac=subset_frac,
        random_seed=random_seed
    )
    
    # Split indices
    np.random.seed(random_seed)
    indices = np.random.permutation(len(full_dataset))
    val_size = int(len(full_dataset) * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    train_dataset.dataset.transform = get_thermal_transforms(image_size, is_train=True)
    val_dataset.dataset.transform = get_thermal_transforms(image_size, is_train=False)
    
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
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'image_size': image_size,
        'in_channels': 1
    }
    
    return train_loader, val_loader, info