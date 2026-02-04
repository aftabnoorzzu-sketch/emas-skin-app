"""
Evaluation script for E-MAS model.
Computes comprehensive metrics and generates evaluation reports.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

from models.emas import create_emas_model
from data.datasets import HAM10000Dataset, PH2Dataset, CombinedDataset, get_transforms


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_model(checkpoint_path, device):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model and checkpoint info
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    num_classes = checkpoint.get('num_classes', 7)
    dataset_type = checkpoint.get('dataset', 'ham10000')
    class_names = checkpoint.get('class_names', None)
    
    # Create model
    model = create_emas_model(num_classes=num_classes, pretrained=False, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, {
        'num_classes': num_classes,
        'dataset': dataset_type,
        'class_names': class_names,
        'epoch': checkpoint.get('epoch', None),
        'val_acc': checkpoint.get('val_acc', None)
    }


def create_test_loader(dataset_type, data_dir, ham_csv, batch_size=32, input_size=224):
    """Create test data loader."""
    transform = get_transforms(input_size=input_size, augment=False)
    
    if dataset_type == 'ham10000':
        dataset = HAM10000Dataset(data_dir, ham_csv, transform=transform, split='test')
        num_classes = 7
    elif dataset_type == 'ph2':
        dataset = PH2Dataset(data_dir, transform=transform, split='test')
        num_classes = 3
    elif dataset_type == 'combined':
        if not isinstance(data_dir, dict):
            raise ValueError("For combined dataset, data_dir must be dict")
        dataset = CombinedDataset(data_dir['ham10000'], data_dir['ph2'], ham_csv, transform=transform)
        num_classes = 7
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return loader, num_classes, dataset.class_to_idx


def compute_metrics(y_true, y_pred, y_prob, num_classes, class_names):
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        num_classes: Number of classes
        class_names: Dictionary mapping indices to class names
        
    Returns:
        Dictionary of metrics
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # ROC-AUC (one-vs-rest)
    try:
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        roc_auc = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='weighted')
        roc_auc_per_class = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average=None)
    except:
        roc_auc = 0.0
        roc_auc_per_class = [0.0] * num_classes
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    target_names = [class_names.get(i, f'Class_{i}') for i in range(num_classes)]
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    return {
        'overall': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc)
        },
        'per_class': {
            class_names.get(i, f'Class_{i}'): {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i]),
                'roc_auc': float(roc_auc_per_class[i]) if i < len(roc_auc_per_class) else 0.0,
                'support': int(report[class_names.get(i, f'Class_{i}')]['support'])
            }
            for i in range(num_classes)
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(y_true, y_prob, num_classes, class_names, save_path):
    """Plot ROC curves for each class."""
    from sklearn.metrics import roc_curve
    
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each class
    for i in range(num_classes):
        if i < y_prob.shape[1]:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            class_name = class_names.get(i, f'Class_{i}')
            plt.plot(fpr, tpr, label=f'{class_name}')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_model(model, dataloader, device, num_classes):
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        device: Device to run on
        num_classes: Number of classes
        
    Returns:
        Predictions, labels, and probabilities
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def save_results_csv(results, class_names, save_path):
    """Save results to CSV format."""
    rows = []
    
    # Overall metrics
    rows.append({
        'Class': 'Overall',
        'Accuracy': results['overall']['accuracy'],
        'Precision': results['overall']['precision'],
        'Recall': results['overall']['recall'],
        'F1-Score': results['overall']['f1_score'],
        'ROC-AUC': results['overall']['roc_auc']
    })
    
    # Per-class metrics
    for class_idx, class_name in class_names.items():
        if class_name in results['per_class']:
            class_results = results['per_class'][class_name]
            rows.append({
                'Class': class_name,
                'Accuracy': '',
                'Precision': class_results['precision'],
                'Recall': class_results['recall'],
                'F1-Score': class_results['f1_score'],
                'ROC-AUC': class_results['roc_auc'],
                'Support': class_results['support']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)


def evaluate(args):
    """Main evaluation function."""
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model, model_info = load_model(args.checkpoint, device)
    print(f"Model loaded (trained for {model_info['epoch']} epochs)")
    print(f"Validation accuracy: {model_info['val_acc']:.4f}")
    
    # Create test loader
    print(f"\nLoading test dataset...")
    if args.dataset == 'combined':
        import json
        data_dir = json.loads(args.data_dir)
    else:
        data_dir = args.data_dir
    
    test_loader, num_classes, class_to_idx = create_test_loader(
        args.dataset, data_dir, args.ham_csv, args.batch_size, args.input_size
    )
    
    # Create reverse mapping
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Number of classes: {num_classes}")
    
    # Evaluate
    print("\nEvaluating model...")
    y_true, y_pred, y_prob = evaluate_model(model, test_loader, device, num_classes)
    
    # Compute metrics
    print("\nComputing metrics...")
    results = compute_metrics(y_true, y_pred, y_prob, num_classes, idx_to_class)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {results['overall']['accuracy']:.4f} ({results['overall']['accuracy']*100:.2f}%)")
    print(f"  Precision: {results['overall']['precision']:.4f}")
    print(f"  Recall:    {results['overall']['recall']:.4f}")
    print(f"  F1-Score:  {results['overall']['f1_score']:.4f}")
    print(f"  ROC-AUC:   {results['overall']['roc_auc']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    for class_name, metrics in results['per_class'].items():
        print(f"  {class_name}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1-Score:  {metrics['f1_score']:.4f}")
        print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"    Support:   {metrics['support']}")
    
    # Create reports directory
    os.makedirs(args.report_dir, exist_ok=True)
    
    # Save JSON report
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    json_path = os.path.join(args.report_dir, f"evaluation_report_{args.dataset}_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON report saved to: {json_path}")
    
    # Save CSV report
    csv_path = os.path.join(args.report_dir, f"evaluation_report_{args.dataset}_{timestamp}.csv")
    save_results_csv(results, idx_to_class, csv_path)
    print(f"CSV report saved to: {csv_path}")
    
    # Plot confusion matrix
    cm_path = os.path.join(args.report_dir, f"confusion_matrix_{args.dataset}_{timestamp}.png")
    class_names_list = [idx_to_class[i] for i in range(num_classes)]
    plot_confusion_matrix(np.array(results['confusion_matrix']), class_names_list, cm_path)
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Plot ROC curves
    roc_path = os.path.join(args.report_dir, f"roc_curves_{args.dataset}_{timestamp}.png")
    plot_roc_curves(y_true, y_prob, num_classes, idx_to_class, roc_path)
    print(f"ROC curves saved to: {roc_path}")
    
    print("\nEvaluation completed!")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate E-MAS model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['ham10000', 'ph2', 'combined'],
                       help='Dataset to evaluate on')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--ham-csv', type=str, default=None,
                       help='Path to HAM10000 metadata CSV')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--input-size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--report-dir', type=str, default='reports',
                       help='Directory to save reports')
    
    args = parser.parse_args()
    
    # Evaluate
    evaluate(args)


if __name__ == "__main__":
    main()
