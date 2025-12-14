#!/usr/bin/env python3
"""
Demo script for geoprivacy attribution heatmaps.

Usage:
    python demo_attribution.py --checkpoint path/to/model.ckpt --image path/to/image.jpg --output output.png
"""

import argparse
from pathlib import Path
import torch
from PIL import Image

from models import BinaryClassifier
from models.explanations import GeoAttributionHeatmap, save_attribution_visualization


def main():
    parser = argparse.ArgumentParser(
        description='Generate geoprivacy attribution heatmap for an image'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained BinaryClassifier checkpoint (.ckpt file)'
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
        default=None,
        help='Path to save output visualization (default: same name as input with _attribution.png suffix)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to run on (default: cuda if available, else cpu)'
    )
    parser.add_argument(
        '--colormap',
        type=str,
        default='jet',
        choices=['jet', 'hot', 'viridis', 'plasma', 'inferno', 'magma'],
        help='Colormap for heatmap visualization (default: jet)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Transparency of heatmap overlay (0=invisible, 1=opaque, default: 0.5)'
    )
    parser.add_argument(
        '--no-smooth',
        action='store_true',
        help='Disable Gaussian smoothing of heatmap'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='gradient',
        choices=['gradient', 'integrated'],
        help='Attribution method: gradient (fast) or integrated (slower but more stable)'
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Set output path
    if args.output is None:
        output_path = image_path.parent / f"{image_path.stem}_attribution.png"
    else:
        output_path = Path(args.output)
    
    print("=" * 60)
    print("Geoprivacy Attribution Heatmap Generator")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input image: {image_path}")
    print(f"Output path: {output_path}")
    print(f"Device: {args.device}")
    print(f"Method: {args.method}")
    print(f"Colormap: {args.colormap}")
    print(f"Alpha: {args.alpha}")
    print(f"Smoothing: {'disabled' if args.no_smooth else 'enabled'}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = BinaryClassifier.load_from_checkpoint(
        str(checkpoint_path),
        strict=False
    )
    model.eval()
    print(f"Model loaded: {model.hparams.arch_name}")
    print(f"Threshold: {model.hparams.threshold_km} km")
    
    # Create attribution system
    print("\nInitializing attribution system...")
    attributor = GeoAttributionHeatmap(
        model=model,
        transform=model.transform,
        device=args.device,
        smooth_heatmap=not args.no_smooth
    )
    
    # Load image
    print(f"\nLoading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    print(f"Image size: {image.size}")
    
    # Generate attribution
    print(f"\nComputing {args.method} attribution...")
    if args.method == 'gradient':
        heatmap, risk_score = attributor(image, return_risk_score=True)
    else:  # integrated gradients
        heatmap, risk_score = attributor.integrated_gradients(
            image, 
            n_steps=50, 
            return_risk_score=True
        )
    
    # Extract values
    heatmap = heatmap[0]  # Remove batch dimension
    risk_score = risk_score[0].item()
    
    print(f"\n{'=' * 60}")
    print(f"RISK SCORE: {risk_score:.4f}")
    print(f"Interpretation: Probability that geolocation error ≤ {model.hparams.threshold_km} km")
    print(f"{'=' * 60}")
    
    # Save visualization
    print(f"\nSaving visualization to: {output_path}")
    save_attribution_visualization(
        image=image,
        heatmap=heatmap,
        save_path=str(output_path),
        risk_score=risk_score,
        colormap=args.colormap,
        alpha=args.alpha,
        dpi=150
    )
    
    print("\n✓ Done!")
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
