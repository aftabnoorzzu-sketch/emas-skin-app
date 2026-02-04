"""
Training script for E-MAS model.
Supports HAM10000, PH2, and Combined datasets with stratified splits and 5-fold CV.
"""

import os
import argparse
import json
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from models.emas import create_emas_model
from data.datasets import (
    HAM10000Dataset, PH2Dataset, CombinedDataset,
    get_transforms, stratified_split, create_stratified_kfold
)


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def create_dataloaders(dataset_type, data_dir, ham_csv, batch_size, input_size=224, 
                       split_mode='holdout', fold_idx=None, num_workers=4):
    """
    Create train/val/test data loaders.
    
    Args:
        dataset_type: 'ham10000', 'ph2', or 'combined'
        data_dir: Path to dataset directory
        ham_csv: Path to HAM10000 CSV file
        batch_size: Batch size
        input_size: Input image size
        split_mode: 'holdout' or '5fold'
        fold_idx: Fold index for 5-fold CV
        num_workers: Number of data loading workers
        
    Returns:
        Dictionary with data loaders and metadata
    """
    # Create transforms
    train_transform = get_transforms(input_size=input_size, augment=True)
    val_transform = get_transforms(input_size=input_size, augment=False)
    
    # Create base datasets
    if dataset_type == 'ham10000':
        train_dataset = HAM10000Dataset(data_dir, ham_csv, transform=train_transform)
        val_dataset = HAM10000Dataset(data_dir, ham_csv, transform=val_transform)
        test_dataset = HAM10000Dataset(data_dir, ham_csv, transform=val_transform)
        num_classes = 7
        class_names = train_dataset.class_to_idx
    
    elif dataset_type == 'ph2':
        train_dataset = PH2Dataset(data_dir, transform=train_transform)
        val_dataset = PH2Dataset(data_dir, transform=val_transform)
        test_dataset = PH2Dataset(data_dir, transform=val_transform)
        num_classes = 3
        class_names = train_dataset.class_to_idx
    
    elif dataset_type == 'combined':
        if not isinstance(data_dir, dict):
            raise ValueError("For combined dataset, data_dir must be dict with 'ham10000' and 'ph2' keys")
        train_dataset = CombinedDataset(data_dir['ham10000'], data_dir['ph2'], ham_csv, transform=train_transform)
        val_dataset = CombinedDataset(data_dir['ham10000'], data_dir['ph2'], ham_csv, transform=val_transform)
        test_dataset = CombinedDataset(data_dir['ham10000'], data_dir['ph2'], ham_csv, transform=val_transform)
        num_classes = 7
        class_names = train_dataset.class_to_idx
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Get labels for stratified splitting
    if hasattr(train_dataset, 'data'):
        labels = [train_dataset.class_to_idx[item['label']] for item in train_dataset.data]
    else:
        labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    
    labels = np.array(labels)
    indices = np.arange(len(labels))
    
    # Create splits
    if split_mode == 'holdout':
        # 70% train, 15% val, 15% test
        from sklearn.model_selection import train_test_split
        
        train_indices, temp_indices = train_test_split(
            indices, test_size=0.3, stratify=labels, random_state=42
        )
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=0.5, stratify=labels[temp_indices], random_state=42
        )
    
    elif split_mode == '5fold':
        folds = create_stratified_kfold(train_dataset, n_splits=5)
        if fold_idx is None:
            fold_idx = 0
        train_indices, val_indices = folds[fold_idx]
        # Use different fold for test
        _, test_indices = folds[(fold_idx + 1) % 5]
    
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")
    
    # Create subset datasets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
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


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=7, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop


class TrainingLogger:
    """Logger for training metrics."""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'learning_rate': []
        }
    
    def log(self, epoch, train_loss, train_acc, val_loss, val_acc, val_f1, lr):
        """Log metrics for an epoch."""
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['val_f1'].append(val_f1)
        self.history['learning_rate'].append(lr)
        
        print(f"Epoch {epoch}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}, "
              f"LR={lr:.6f}")
    
    def save(self, path):
        """Save training history to JSON."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_best_epoch(self, metric='val_acc'):
        """Get epoch with best metric."""
        if metric in ['val_acc', 'val_f1']:
            best_idx = np.argmax(self.history[metric])
        else:
            best_idx = np.argmin(self.history[metric])
        return best_idx + 1, self.history[metric][best_idx]


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device, num_classes):
    """Evaluate model on a dataset."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    # Calculate additional metrics
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # ROC-AUC (one-vs-rest)
    try:
        labels_bin = label_binarize(all_labels, classes=range(num_classes))
        roc_auc = roc_auc_score(labels_bin, np.array(all_probs), multi_class='ovr', average='weighted')
    except:
        roc_auc = 0.0
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def train(args):
    """Main training function."""
    
    # Set seeds for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create data loaders
    print(f"\nLoading {args.dataset} dataset...")
    data_info = create_dataloaders(
        dataset_type=args.dataset,
        data_dir=args.data_dir,
        ham_csv=args.ham_csv,
        batch_size=args.batch_size,
        input_size=args.input_size,
        split_mode=args.split_mode,
        fold_idx=args.fold_idx,
        num_workers=args.num_workers
    )
    
    print(f"Train size: {data_info['train_size']}")
    print(f"Val size: {data_info['val_size']}")
    print(f"Test size: {data_info['test_size']}")
    print(f"Number of classes: {data_info['num_classes']}")
    
    # Create model
    print("\nCreating E-MAS model...")
    model = create_emas_model(
        num_classes=data_info['num_classes'],
        pretrained=True,
        device=device
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, mode='max')
    
    # Training logger
    logger = TrainingLogger()
    
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0.0
    best_model_path = None
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, data_info['train_loader'], criterion, optimizer, device
        )
        
        # Validate
        val_results = evaluate(
            model, data_info['val_loader'], criterion, device, data_info['num_classes']
        )
        
        # Update scheduler
        scheduler.step(val_results['accuracy'])
        
        # Log
        logger.log(
            epoch, train_loss, train_acc,
            val_results['loss'], val_results['accuracy'], val_results['f1'],
            optimizer.param_groups[0]['lr']
        )
        
        # Save best model
        if val_results['accuracy'] > best_val_acc:
            best_val_acc = val_results['accuracy']
            
            # Save checkpoint
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            best_model_path = os.path.join(
                args.checkpoint_dir,
                f"emas_{args.dataset}_best_{timestamp}.pth"
            )
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'num_classes': data_info['num_classes'],
                'dataset': args.dataset,
                'class_names': data_info['class_names']
            }, best_model_path)
            
            print(f"  -> Saved best model: {best_model_path}")
        
        # Early stopping check
        if early_stopping(val_results['accuracy']):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
        
        epoch_time = time.time() - start_time
        print(f"  Epoch time: {epoch_time:.2f}s")
    
    # Save training history
    history_path = os.path.join(args.checkpoint_dir, f"training_history_{args.dataset}.json")
    logger.save(history_path)
    print(f"\nTraining history saved to: {history_path}")
    
    # Final evaluation on test set
    if best_model_path and os.path.exists(best_model_path):
        print("\nEvaluating best model on test set...")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_results = evaluate(
            model, data_info['test_loader'], criterion, device, data_info['num_classes']
        )
        
        print("\nTest Set Results:")
        print(f"  Accuracy:  {test_results['accuracy']:.4f}")
        print(f"  Precision: {test_results['precision']:.4f}")
        print(f"  Recall:    {test_results['recall']:.4f}")
        print(f"  F1-Score:  {test_results['f1']:.4f}")
        print(f"  ROC-AUC:   {test_results['roc_auc']:.4f}")
        
        # Save test results
        test_results_path = os.path.join(args.checkpoint_dir, f"test_results_{args.dataset}.json")
        with open(test_results_path, 'w') as f:
            json.dump({
                'accuracy': test_results['accuracy'],
                'precision': test_results['precision'],
                'recall': test_results['recall'],
                'f1': test_results['f1'],
                'roc_auc': test_results['roc_auc'],
                'best_model_path': best_model_path
            }, f, indent=2)
        
        print(f"\nTest results saved to: {test_results_path}")
    
    print("\nTraining completed!")
    return best_model_path


def main():
    parser = argparse.ArgumentParser(description='Train E-MAS model')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['ham10000', 'ph2', 'combined'],
                       help='Dataset to use')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--ham-csv', type=str, default=None,
                       help='Path to HAM10000 metadata CSV')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=40,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--input-size', type=int, default=224,
                       help='Input image size')
    
    # Split arguments
    parser.add_argument('--split-mode', type=str, default='holdout',
                       choices=['holdout', '5fold'],
                       help='Data splitting mode')
    parser.add_argument('--fold-idx', type=int, default=None,
                       help='Fold index for 5-fold CV (0-4)')
    
    # Other arguments
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--early-stopping-patience', type=int, default=7,
                       help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Handle combined dataset data_dir
    if args.dataset == 'combined':
        import json as json_mod
        args.data_dir = json_mod.loads(args.data_dir)
    
    # Train
    train(args)


if __name__ == "__main__":
    main()
