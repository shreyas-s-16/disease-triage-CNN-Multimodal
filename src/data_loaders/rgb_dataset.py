"""
RGB (HAM10000) Dataset Loader
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from typing import Optional, Tuple, List
import numpy as np


class HAM10000Dataset(Dataset):
    """HAM10000 RGB skin lesion dataset."""
    
    CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
    
    def __init__(
        self,
        metadata_path: str,
        images_dir: str,
        transform: Optional[transforms.Compose] = None,
        subset_frac: float = 1.0,
        random_seed: int = 42,
        balance_classes: bool = False
    ):
        """
        Args:
            metadata_path: Path to HAM10000_metadata.csv
            images_dir: Directory containing image files
            transform: Image transforms
            subset_frac: Fraction of data to use (for fast pass)
            random_seed: Random seed for reproducibility
            balance_classes: Whether to balance classes in subset
        """
        self.metadata_path = metadata_path
        self.images_dir = images_dir
        self.transform = transform
        
        # Load metadata
        self.df = pd.read_csv(metadata_path)
        
        # Map diagnosis to class index
        self.df['label'] = self.df['dx'].map(self.CLASS_TO_IDX)
        
        # Filter to only images that exist (check both part directories)
        def find_image_path(image_id):
            for part_dir in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
                path = os.path.join(images_dir, part_dir, f"{image_id}.jpg")
                if os.path.exists(path):
                    return path
            return os.path.join(images_dir, f"{image_id}.jpg")  # fallback
        
        self.df['image_path'] = self.df['image_id'].apply(find_image_path)
        self.df = self.df[self.df['image_path'].apply(os.path.exists)].reset_index(drop=True)
        
        print(f"Found {len(self.df)} images with valid paths")
        
        # Apply subset
        if subset_frac < 1.0:
            if balance_classes:
                # Balanced sampling
                dfs = []
                for cls in self.CLASSES:
                    cls_df = self.df[self.df['dx'] == cls]
                    n_samples = max(1, int(len(cls_df) * subset_frac))
                    dfs.append(cls_df.sample(n=min(n_samples, len(cls_df)), random_state=random_seed))
                self.df = pd.concat(dfs).reset_index(drop=True)
            else:
                self.df = self.df.sample(frac=subset_frac, random_state=random_seed).reset_index(drop=True)
        
        print(f"Dataset size: {len(self.df)}")
        print(f"Class distribution:\n{self.df['dx'].value_counts()}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_path = row['image_path']
        label = row['label']
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for balanced sampling."""
        class_counts = self.df['label'].value_counts().sort_index()
        total = len(self.df)
        # Handle missing classes
        weights_dict = {cls: total / (len(self.CLASSES) * class_counts.get(cls, 1)) for cls in range(len(self.CLASSES))}
        return torch.tensor([weights_dict[i] for i in range(len(self.CLASSES))], dtype=torch.float32)
    
    def get_sampler(self) -> WeightedRandomSampler:
        """Get weighted sampler for balanced training."""
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[label] for label in self.df['label']]
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def get_rgb_transforms(image_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """Get standard transforms for RGB images."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def create_rgb_dataloaders(
    metadata_path: str,
    images_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    subset_frac: float = 1.0,
    balance_classes: bool = False,
    val_split: float = 0.2,
    num_workers: int = 4,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, dict]:
    """Create train and validation dataloaders for RGB data."""
    
    # Full dataset
    full_dataset = HAM10000Dataset(
        metadata_path=metadata_path,
        images_dir=images_dir,
        transform=None,  # Will set per split
        subset_frac=subset_frac,
        random_seed=random_seed,
        balance_classes=balance_classes
    )
    
    # Split indices
    np.random.seed(random_seed)
    indices = np.random.permutation(len(full_dataset))
    val_size = int(len(full_dataset) * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # Create subset datasets with appropriate transforms
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Apply transforms
    train_dataset.dataset.transform = get_rgb_transforms(image_size, is_train=True)
    val_dataset.dataset.transform = get_rgb_transforms(image_size, is_train=False)
    
    # Sampler for balanced training (only for training subset)
    sampler = None
    if balance_classes:
        # Create sampler using only training indices
        train_labels = [full_dataset.df.iloc[idx]['label'] for idx in train_indices]
        class_weights = full_dataset.get_class_weights()
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
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
        'num_classes': len(HAM10000Dataset.CLASSES),
        'class_names': HAM10000Dataset.CLASSES,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'class_weights': full_dataset.get_class_weights()
    }
    
    return train_loader, val_loader, info