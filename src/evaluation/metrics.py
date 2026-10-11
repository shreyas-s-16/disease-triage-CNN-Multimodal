"""
Evaluation utilities.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: List[str] = None
) -> Dict:
    """Evaluate model and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            _, preds = logits.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Overall accuracy
    acc = accuracy_score(all_labels, all_preds)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=range(num_classes), zero_division=0
    )
    
    # Macro averages
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    
    # Weighted averages
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    
    results = {
        'accuracy': acc,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist()
        },
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist()
    }
    
    if class_names:
        results['class_names'] = class_names
        results['per_class_dict'] = {
            class_names[i]: {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': int(support[i])
            }
            for i in range(num_classes)
        }
    
    return results


def evaluate_multitask(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_names: List[str],
    class_names_dict: Dict[str, List[str]] = None
) -> Dict:
    """Evaluate multi-task model (e.g., fusion model)."""
    model.eval()
    all_preds = {task: [] for task in task_names}
    all_labels = {task: [] for task in task_names}
    
    with torch.no_grad():
        for batch in loader:
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            outputs = model(batch['rgb'], batch['thermal'], batch['tabular'])
            
            for task in task_names:
                if task == 'fever':
                    preds = (outputs[task] > 0).long()
                else:
                    _, preds = outputs[task].max(1)
                all_preds[task].extend(preds.cpu().numpy())
                all_labels[task].extend(batch[f'{task}_label'].cpu().numpy())
    
    results = {}
    for task in task_names:
        preds = np.array(all_preds[task])
        labels = np.array(all_labels[task])
        
        if task == 'fever':
            num_classes = 2
        else:
            num_classes = len(class_names_dict.get(task, np.unique(labels))) if class_names_dict else len(np.unique(labels))
        
        acc = accuracy_score(labels, preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average=None, labels=range(num_classes), zero_division=0
        )
        
        results[task] = {
            'accuracy': acc,
            'macro_precision': precision.mean(),
            'macro_recall': recall.mean(),
            'macro_f1': f1.mean(),
            'weighted_precision': np.average(precision, weights=support),
            'weighted_recall': np.average(recall, weights=support),
            'weighted_f1': np.average(f1, weights=support),
            'per_class': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'f1': f1.tolist(),
                'support': support.tolist()
            },
            'confusion_matrix': confusion_matrix(labels, preds, labels=range(num_classes)).tolist()
        }
        
        if class_names_dict and task in class_names_dict:
            results[task]['class_names'] = class_names_dict[task]
            results[task]['per_class_dict'] = {
                class_names_dict[task][i]: {
                    'precision': precision[i],
                    'recall': recall[i],
                    'f1': f1[i],
                    'support': int(support[i])
                }
                for i in range(min(num_classes, len(class_names_dict[task])))
            }
    
    return results


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, save_path: str = None):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_history(history: Dict, save_path: str = None):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def save_evaluation_report(results: Dict, save_dir: str, prefix: str = ""):
    """Save evaluation results to files."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics as JSON
    import json
    with open(save_dir / f"{prefix}metrics.json", 'w') as f:
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        json.dump(convert(results), f, indent=2)
    
    # Print summary
    print(f"\n=== {prefix} Evaluation Results ===")
    if 'accuracy' in results:
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Macro F1: {results['macro_f1']:.4f}")
        print(f"Weighted F1: {results['weighted_f1']:.4f}")
        if 'per_class_dict' in results:
            for cls, metrics in results['per_class_dict'].items():
                print(f"  {cls}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    else:
        for task, metrics in results.items():
            print(f"\n{task.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Macro F1: {metrics['macro_f1']:.4f}")
            if 'per_class_dict' in metrics:
                for cls, m in metrics['per_class_dict'].items():
                    print(f"    {cls}: P={m['precision']:.3f}, R={m['recall']:.3f}, F1={m['f1']:.3f}")


def run_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_dir: str,
    prefix: str = "",
    class_names: List[str] = None,
    is_multitask: bool = False,
    task_names: List[str] = None,
    class_names_dict: Dict[str, List[str]] = None
):
    """Run full evaluation and save results."""
    if is_multitask:
        results = evaluate_multitask(model, loader, device, task_names, class_names_dict)
    else:
        results = evaluate_model(model, loader, device, len(class_names) if class_names else 10, class_names)
    
    save_evaluation_report(results, save_dir, prefix)
    
    # Plot confusion matrices
    if is_multitask:
        for task, metrics in results.items():
            if 'confusion_matrix' in metrics and 'class_names' in metrics:
                plot_confusion_matrix(
                    np.array(metrics['confusion_matrix']),
                    metrics['class_names'],
                    f"{prefix} {task} Confusion Matrix",
                    str(Path(save_dir) / f"{prefix}{task}_cm.png")
                )
    else:
        if 'confusion_matrix' in results and class_names:
            plot_confusion_matrix(
                np.array(results['confusion_matrix']),
                class_names,
                f"{prefix} Confusion Matrix",
                str(Path(save_dir) / f"{prefix}_cm.png")
            )
    
    return results