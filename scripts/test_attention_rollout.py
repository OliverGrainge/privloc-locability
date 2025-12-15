#!/usr/bin/env python3
"""
Test script for attention rollout visualization.

This script loads a trained BinaryClassifier model and generates
attention rollout saliency maps for input images.

Usage:
    python scripts/test_attention_rollout.py \
        --checkpoint path/to/model.ckpt \
        --image path/to/image.jpg \
        --output attention_rollout.png \
        --arch_name megaloc
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from models.binaryclassifer import BinaryClassifier
from models.binaryclassifer.archs import load_arch_and_transform
from models.explanations.attribution import GeoAttributionHeatmap


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test attention rollout visualization on an image'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.ckpt file)'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='attention_rollout.png',
        help='Path to save output visualization (default: attention_rollout.png)'
    )
    parser.add_argument(
        '--arch_name',
        type=str,
        default='megaloc',
        help='Architecture name (default: megaloc)'
    )
    parser.add_argument(
        '--head_fusion',
        type=str,
        default='mean',
        choices=['mean', 'max', 'min'],
        help='Method to fuse attention heads (default: mean)'
    )
    parser.add_argument(
        '--discard_ratio',
        type=float,
        default=0.1,
        help='Ratio of lowest attentions to discard (default: 0.1)'
    )
    parser.add_argument(
        '--colormap',
        type=str,
        default='jet',
        help='Matplotlib colormap for visualization (default: jet)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Alpha blending for overlay (0=only image, 1=only heatmap, default: 0.5)'
    )
    
    return parser.parse_args()


def load_model(checkpoint_path: str, arch_name: str) -> BinaryClassifier:
    """
    Load BinaryClassifier model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        arch_name: Architecture name
        
    Returns:
        Loaded model
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load model from checkpoint
    model = BinaryClassifier.load_from_checkpoint(
        checkpoint_path,
        arch_name=arch_name,
        strict=False
    )
    
    model.eval()
    print("✓ Model loaded successfully")
    
    return model


def load_image(image_path: str) -> Image.Image:
    """
    Load image from file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        PIL Image
    """
    print(f"Loading image: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    print(f"✓ Image loaded successfully (size: {image.size})")
    
    return image


def visualize_attention(
    image: Image.Image,
    saliency: torch.Tensor,
    output_path: str,
    colormap: str = 'jet',
    alpha: float = 0.5
):
    """
    Create and save attention visualization.
    
    Args:
        image: Original PIL image
        saliency: Saliency map tensor [H, W]
        output_path: Path to save visualization
        colormap: Matplotlib colormap name
        alpha: Alpha blending value
    """
    print(f"\nCreating visualization...")
    
    # Convert saliency to numpy
    if isinstance(saliency, torch.Tensor):
        saliency_np = saliency.cpu().numpy()
    else:
        saliency_np = saliency
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Attention heatmap only
    im = axes[1].imshow(saliency_np, cmap=colormap)
    axes[1].set_title('Attention Rollout', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Overlay on original image
    axes[2].imshow(image)
    # Resize image to match saliency size for overlay
    image_resized = np.array(image.resize((saliency_np.shape[1], saliency_np.shape[0])))
    axes[2].imshow(image_resized)
    axes[2].imshow(saliency_np, cmap=colormap, alpha=alpha)
    axes[2].set_title('Overlay', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    
    # Also print some statistics
    print(f"\nSaliency statistics:")
    print(f"  Shape: {saliency_np.shape}")
    print(f"  Min: {saliency_np.min():.4f}")
    print(f"  Max: {saliency_np.max():.4f}")
    print(f"  Mean: {saliency_np.mean():.4f}")
    print(f"  Std: {saliency_np.std():.4f}")
    
    plt.close()


def main():
    """Main function."""
    args = parse_args()
    
    # Validate paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print("="*60)
    print("ATTENTION ROLLOUT VISUALIZATION TEST")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Image: {image_path}")
    print(f"Output: {args.output}")
    print(f"Architecture: {args.arch_name}")
    print(f"Head fusion: {args.head_fusion}")
    print(f"Discard ratio: {args.discard_ratio}")
    print("="*60)
    print()
    
    # Load model
    model = load_model(str(checkpoint_path), args.arch_name)
    
    # Get transform
    _, transform = load_arch_and_transform(args.arch_name)
    
    # Create attribution object
    print("\nInitializing GeoAttributionHeatmap...")
    attributor = GeoAttributionHeatmap(model, transform)
    print("✓ Attribution object created")
    
    # Load image
    image = load_image(str(image_path))
    
    # Compute attention rollout
    print(f"\nComputing attention rollout...")
    print(f"  (This may take a moment...)")
    
    with torch.no_grad():
        saliency = attributor(
            image,
            head_fusion=args.head_fusion,
            discard_ratio=args.discard_ratio
        )
    
    print("✓ Attention rollout computed successfully")
    
    # Create visualization
    visualize_attention(
        image,
        saliency,
        args.output,
        colormap=args.colormap,
        alpha=args.alpha
    )
    
    # Get model prediction
    print("\nModel prediction:")
    image_tensor = transform(image).unsqueeze(0).to(attributor.device)
    with torch.no_grad():
        logit = model(image_tensor)
        prob = torch.sigmoid(logit).item()
    
    print(f"  Logit: {logit.item():.4f}")
    print(f"  Probability: {prob:.4f}")
    print(f"  Prediction: {'Positive' if prob > 0.5 else 'Negative'} (threshold=0.5)")
    
    print("\n" + "="*60)
    print("✓ Test completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
