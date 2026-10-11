"""
Evaluate RGB baseline model and generate evaluation artifacts.
"""
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from src.utils import get_device
from src.data_loaders import create_rgb_dataloaders, HAM10000Dataset, get_rgb_transforms
from src.models import RGBEncoder
from src.evaluation import run_evaluation, evaluate_model, plot_confusion_matrix, plot_training_history


def generate_gradcam(model, images, target_layer, class_idx, device):
    """Generate Grad-CAM heatmap for a given image and class."""
    model.eval()
    
    # Forward hook to capture gradients
    gradients = []
    activations = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # Register hooks
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward pass
    images = images.to(device)
    output = model(images)
    
    # Backward pass for target class
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, class_idx] = 1
    output.backward(gradient=one_hot, retain_graph=True)
    
    # Remove hooks
    handle_fwd.remove()
    handle_bwd.remove()
    
    # Generate CAM
    grad = gradients[0].cpu().data.numpy()[0]
    act = activations[0].cpu().data.numpy()[0]
    
    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * act[i]
    
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = np.uint8(255 * cam)
    
    return cam


def main(config_path: str, checkpoint_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = get_device()
    
    # Load full dataset with test split
    splits_path = Path(config['data'].get('splits_path', 'data/rgb/processed/splits.csv'))
    if splits_path.exists():
        import pandas as pd
        splits_df = pd.read_csv(splits_path)
        test_idx = splits_df[splits_df['split'] == 'test']['index'].values
    else:
        # Fallback to val split
        test_idx = None
    
    full_dataset = HAM10000Dataset(
        metadata_path=config['data']['metadata_path'],
        images_dir=config['data']['images_dir'],
        transform=None,
        subset_frac=1.0,
        random_seed=config.get('seed', 42),
        balance_classes=False
    )
    
    if test_idx is not None:
        test_dataset = Subset(full_dataset, test_idx)
    else:
        # Use val split
        val_split = config['data'].get('val_split', 0.2)
        np.random.seed(config.get('seed', 42))
        indices = np.random.permutation(len(full_dataset))
        val_size = int(len(full_dataset) * val_split)
        test_idx = indices[:val_size]
        test_dataset = Subset(full_dataset, test_idx)
    
    test_transform = get_rgb_transforms(config['data']['image_size'], is_train=False)
    test_dataset.dataset.transform = test_transform
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
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
    
    # Run evaluation
    results = run_evaluation(
        model=model,
        loader=test_loader,
        device=device,
        save_dir="results/rgb",
        prefix="rgb_",
        class_names=info['class_names']
    )
    
    # Plot confusion matrix
    cm = np.array(results['confusion_matrix'])
    plot_confusion_matrix(cm, info['class_names'], "RGB Test Confusion Matrix", 
                          "results/rgb/confusion_matrix.png")
    
    # Plot training history if available
    history_path = Path(config['training']['checkpoint_dir']) / 'history.npy'
    if history_path.exists():
        history = np.load(history_path, allow_pickle=True).item()
        plot_training_history(history, "results/rgb/training_curves.png")
    
    # Generate Grad-CAM for 12 test images (correct and incorrect)
    print("Generating Grad-CAM visualizations...")
    gradcam_dir = Path("results/rgb/gradcam")
    gradcam_dir.mkdir(parents=True, exist_ok=True)
    
    # Get target layer (last conv layer of EfficientNet-B0)
    target_layer = model.backbone.features[-1]
    
    correct_images = []
    incorrect_images = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            _, preds = logits.max(1)
            
            for i in range(min(images.size(0), 4)):  # Collect up to 4 per batch
                if len(correct_images) < 6 and preds[i] == labels[i]:
                    correct_images.append((images[i].cpu(), labels[i].item(), preds[i].item()))
                elif len(incorrect_images) < 6 and preds[i] != labels[i]:
                    incorrect_images.append((images[i].cpu(), labels[i].item(), preds[i].item()))
                
                if len(correct_images) >= 6 and len(incorrect_images) >= 6:
                    break
            
            if len(correct_images) >= 6 and len(incorrect_images) >= 6:
                break
    
    # Generate Grad-CAM for collected images
    all_images = correct_images + incorrect_images
    for idx, (img, true_label, pred_label) in enumerate(all_images):
        img_batch = img.unsqueeze(0).to(device)
        cam = generate_gradcam(model, img_batch, target_layer, pred_label, device)
        
        # Resize CAM to image size
        cam_resized = Image.fromarray(cam).resize((config['data']['image_size'], config['data']['image_size']), Image.BILINEAR)
        cam_resized = np.array(cam_resized)
        
        # Original image
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        
        # Overlay heatmap
        heatmap = plt.cm.jet(cam_resized / 255.0)[:, :, :3]
        overlay = (0.5 * img_np + 0.5 * (heatmap * 255)).astype(np.uint8)
        
        # Save
        status = "correct" if true_label == pred_label else "incorrect"
        save_path = gradcam_dir / f"gradcam_{idx}_{status}_true{true_label}_pred{pred_label}.png"
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_np)
        axes[0].set_title(f"Original\nTrue: {info['class_names'][true_label]}")
        axes[0].axis('off')
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title(f"Grad-CAM\nPred: {info['class_names'][pred_label]}")
        axes[1].axis('off')
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay ({status})")
        axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Create sample predictions grid
    print("Creating sample predictions grid...")
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (img, true_label, pred_label) in enumerate(all_images[:12]):
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        
        axes[idx].imshow(img_np)
        status = "✓" if true_label == pred_label else "✗"
        color = 'green' if true_label == pred_label else 'red'
        axes[idx].set_title(
            f"{status} True: {info['class_names'][true_label]}\nPred: {info['class_names'][pred_label]}",
            color=color, fontsize=10
        )
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig("results/rgb/sample_predictions.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("RGB evaluation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)