"""
Dataset loaders for HAM10000 and PH2 dermoscopic image datasets.
Supports both folder structure and CSV-based loading with stratified splits.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Dataset class names
HAM10000_CLASSES = [
    'akiec',  # Actinic keratoses
    'bcc',    # Basal cell carcinoma
    'bkl',    # Benign keratosis-like lesions
    'df',     # Dermatofibroma
    'mel',    # Melanoma
    'nv',     # Melanocytic nevi
    'vasc'    # Vascular lesions
]

HAM10000_CLASS_NAMES = {
    'akiec': 'Actinic keratoses',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions'
}

PH2_CLASSES = ['nev', 'atypical', 'mel']

PH2_CLASS_NAMES = {
    'nev': 'Nevus',
    'atypical': 'Atypical',
    'mel': 'Melanoma'
}

COMBINED_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc', 'nev', 'atypical']


class HAM10000Dataset(Dataset):
    """
    HAM10000 dataset loader supporting both folder structure and CSV metadata.
    
    Args:
        root_dir: Root directory containing images
        csv_path: Path to metadata CSV file (optional)
        transform: PyTorch transforms for preprocessing
        split: Data split ('train', 'val', 'test')
    """
    def __init__(self, root_dir, csv_path=None, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # Load data
        if csv_path and os.path.exists(csv_path):
            self.data = self._load_from_csv(csv_path)
        else:
            self.data = self._load_from_folder()
        
        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(HAM10000_CLASSES)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
    def _load_from_csv(self, csv_path):
        """Load dataset from CSV metadata file."""
        df = pd.read_csv(csv_path)
        
        data = []
        for _, row in df.iterrows():
            image_id = row['image_id']
            label = row['dx']
            
            # Find image file
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                path = os.path.join(self.root_dir, image_id + ext)
                if os.path.exists(path):
                    image_path = path
                    break
            
            if image_path:
                data.append({
                    'image_path': image_path,
                    'label': label,
                    'image_id': image_id
                })
        
        return data
    
    def _load_from_folder(self):
        """Load dataset from folder structure (ImageFolder style)."""
        data = []
        
        for class_name in HAM10000_CLASSES:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(class_dir, img_name)
                    data.append({
                        'image_path': image_path,
                        'label': class_name,
                        'image_id': os.path.splitext(img_name)[0]
                    })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        label = item['label']
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label index
        label_idx = self.class_to_idx[label]
        
        return image, label_idx
    
    def get_class_distribution(self):
        """Get distribution of classes in the dataset."""
        labels = [item['label'] for item in self.data]
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))


class PH2Dataset(Dataset):
    """
    PH2 dataset loader supporting folder structure.
    
    Args:
        root_dir: Root directory containing images organized by class
        transform: PyTorch transforms for preprocessing
        split: Data split ('train', 'val', 'test')
    """
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # Load data from folder structure
        self.data = self._load_from_folder()
        
        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(PH2_CLASSES)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
    
    def _load_from_folder(self):
        """Load dataset from folder structure."""
        data = []
        
        for class_name in PH2_CLASSES:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(class_dir, img_name)
                    data.append({
                        'image_path': image_path,
                        'label': class_name,
                        'image_id': os.path.splitext(img_name)[0]
                    })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        label = item['label']
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label index
        label_idx = self.class_to_idx[label]
        
        return image, label_idx
    
    def get_class_distribution(self):
        """Get distribution of classes in the dataset."""
        labels = [item['label'] for item in self.data]
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))


class CombinedDataset(Dataset):
    """
    Combined HAM10000 + PH2 dataset.
    Maps PH2 classes to HAM10000 classes where applicable.
    """
    def __init__(self, ham10000_dir, ph2_dir, ham_csv=None, transform=None, split='train'):
        self.transform = transform
        self.split = split
        
        # Load HAM10000
        ham_dataset = HAM10000Dataset(ham10000_dir, ham_csv, transform=None, split=split)
        
        # Load PH2
        ph2_dataset = PH2Dataset(ph2_dir, transform=None, split=split)
        
        # Combine data with remapped labels
        self.data = []
        
        # Add HAM10000 data
        for item in ham_dataset.data:
            self.data.append({
                'image_path': item['image_path'],
                'label': item['label'],
                'source': 'ham10000'
            })
        
        # Add PH2 data with remapped labels
        # PH2: nev -> nv, atypical -> bkl, mel -> mel
        ph2_to_ham = {
            'nev': 'nv',
            'atypical': 'bkl',
            'mel': 'mel'
        }
        
        for item in ph2_dataset.data:
            remapped_label = ph2_to_ham.get(item['label'], item['label'])
            self.data.append({
                'image_path': item['image_path'],
                'label': remapped_label,
                'source': 'ph2'
            })
        
        # Create class to index mapping (HAM10000 classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(HAM10000_CLASSES)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        label = item['label']
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label index
        label_idx = self.class_to_idx[label]
        
        return image, label_idx
    
    def get_class_distribution(self):
        """Get distribution of classes in the dataset."""
        labels = [item['label'] for item in self.data]
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))


def get_transforms(input_size=224, augment=True):
    """
    Get preprocessing and augmentation transforms.
    
    Args:
        input_size: Input image size (default 224 for E-MAS)
        augment: Whether to apply data augmentation
        
    Returns:
        transforms.Compose object
    """
    # ImageNet normalization values
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomRotation(degrees=20),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(
                size=(input_size, input_size),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize
        ])
    else:
        # Validation/test transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize
        ])
    
    return transform


def stratified_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                     random_state=42):
    """
    Perform stratified train/val/test split on a dataset.
    
    Args:
        dataset: PyTorch Dataset instance
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with indices for each split
    """
    # Get all labels
    if hasattr(dataset, 'data'):
        labels = [dataset.class_to_idx[item['label']] for item in dataset.data]
    else:
        # Fallback: iterate through dataset
        labels = [dataset[i][1] for i in range(len(dataset))]
    
    labels = np.array(labels)
    indices = np.arange(len(labels))
    
    # First split: train vs (val + test)
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=(val_ratio + test_ratio),
        stratify=labels[indices],
        random_state=random_state
    )
    
    # Second split: val vs test
    val_test_labels = labels[temp_indices]
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1 - val_ratio_adjusted),
        stratify=val_test_labels,
        random_state=random_state
    )
    
    return {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }


def create_stratified_kfold(dataset, n_splits=5, random_state=42):
    """
    Create stratified k-fold cross-validation splits.
    
    Args:
        dataset: PyTorch Dataset instance
        n_splits: Number of folds
        random_state: Random seed
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    # Get all labels
    if hasattr(dataset, 'data'):
        labels = [dataset.class_to_idx[item['label']] for item in dataset.data]
    else:
        labels = [dataset[i][1] for i in range(len(dataset))]
    
    labels = np.array(labels)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    folds = []
    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        folds.append((train_idx, val_idx))
    
    return folds


def create_data_loaders(dataset_type='ham10000', data_dir=None, ham_csv=None,
                       batch_size=32, num_workers=4, input_size=224,
                       split_mode='holdout', fold_idx=None):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        dataset_type: 'ham10000', 'ph2', or 'combined'
        data_dir: Path to dataset directory (or dict with 'ham10000' and 'ph2' keys for combined)
        ham_csv: Path to HAM10000 metadata CSV
        batch_size: Batch size
        num_workers: Number of data loading workers
        input_size: Input image size
        split_mode: 'holdout' or '5fold'
        fold_idx: Fold index for 5-fold CV (0-4)
        
    Returns:
        Dictionary containing data loaders and dataset info
    """
    # Create transforms
    train_transform = get_transforms(input_size=input_size, augment=True)
    val_transform = get_transforms(input_size=input_size, augment=False)
    
    # Create dataset
    if dataset_type == 'ham10000':
        full_dataset = HAM10000Dataset(data_dir, ham_csv, transform=None)
        num_classes = len(HAM10000_CLASSES)
        class_names = HAM10000_CLASS_NAMES
    elif dataset_type == 'ph2':
        full_dataset = PH2Dataset(data_dir, transform=None)
        num_classes = len(PH2_CLASSES)
        class_names = PH2_CLASS_NAMES
    elif dataset_type == 'combined':
        if not isinstance(data_dir, dict):
            raise ValueError("For combined dataset, data_dir must be a dict with 'ham10000' and 'ph2' keys")
        full_dataset = CombinedDataset(
            data_dir['ham10000'], 
            data_dir['ph2'], 
            ham_csv, 
            transform=None
        )
        num_classes = len(HAM10000_CLASSES)
        class_names = HAM10000_CLASS_NAMES
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Create splits
    if split_mode == 'holdout':
        splits = stratified_split(full_dataset)
        train_indices = splits['train']
        val_indices = splits['val']
        test_indices = splits['test']
    elif split_mode == '5fold':
        folds = create_stratified_kfold(full_dataset, n_splits=5)
        if fold_idx is None:
            fold_idx = 0
        train_indices, val_indices = folds[fold_idx]
        # Use last fold as test set
        _, test_indices = folds[(fold_idx + 1) % 5]
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")
    
    # Create subset datasets with appropriate transforms
    train_dataset = torch.utils.data.Subset(
        torch.utils.data.TensorDataset(torch.zeros(len(train_indices))), 
        range(len(train_indices))
    )
    
    # Use custom subset that applies transforms
    class TransformedSubset(Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            image, label = self.dataset[real_idx] if hasattr(self.dataset, '__getitem__') else self.dataset.data[real_idx]
            
            # Load image if needed
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
    
    # Re-create datasets with transforms
    if dataset_type == 'ham10000':
        base_train = HAM10000Dataset(data_dir, ham_csv, transform=train_transform)
        base_val = HAM10000Dataset(data_dir, ham_csv, transform=val_transform)
        base_test = HAM10000Dataset(data_dir, ham_csv, transform=val_transform)
    elif dataset_type == 'ph2':
        base_train = PH2Dataset(data_dir, transform=train_transform)
        base_val = PH2Dataset(data_dir, transform=val_transform)
        base_test = PH2Dataset(data_dir, transform=val_transform)
    else:  # combined
        base_train = CombinedDataset(data_dir['ham10000'], data_dir['ph2'], ham_csv, transform=train_transform)
        base_val = CombinedDataset(data_dir['ham10000'], data_dir['ph2'], ham_csv, transform=val_transform)
        base_test = CombinedDataset(data_dir['ham10000'], data_dir['ph2'], ham_csv, transform=val_transform)
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(base_train, train_indices)
    val_dataset = torch.utils.data.Subset(base_val, val_indices)
    test_dataset = torch.utils.data.Subset(base_test, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'num_classes': num_classes,
        'class_names': class_names,
        'train_size': len(train_indices),
        'val_size': len(val_indices),
        'test_size': len(test_indices)
    }


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loaders...")
    
    # Test transforms
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)
    
    print("\nTrain transforms:", train_transform)
    print("\nVal transforms:", val_transform)
    
    # Test class mappings
    print("\nHAM10000 classes:", HAM10000_CLASSES)
    print("HAM10000 class names:", HAM10000_CLASS_NAMES)
    print("\nPH2 classes:", PH2_CLASSES)
    print("PH2 class names:", PH2_CLASS_NAMES)
