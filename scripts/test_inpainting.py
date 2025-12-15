#!/usr/bin/env python3
"""
Test script for inpainting with region extraction.

This script combines Grounding DINO + SAM (region extraction) with 
SDXL inpainting to detect regions and replace them with AI-generated content.

Usage:
    # Remove all people from an image
    python scripts/test_inpainting.py \
        --image path/to/image.jpg \
        --output inpainted.png \
        --detect "person" \
        --prompt "natural park scenery with grass and trees"
    
    # Remove cars and replace with roads
    python scripts/test_inpainting.py \
        --image path/to/image.jpg \
        --detect "car" \
        --prompt "empty road"
    
    # Remove multiple object types
    python scripts/test_inpainting.py \
        --image path/to/image.jpg \
        --detect "person" "car" "sign" \
        --prompt "clean natural landscape"
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from models.explanations.region import EditableRegionExtraction
from models.explanations.inpainting import Inpainting


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test inpainting with automatic region detection'
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
        default='inpainted.png',
        help='Path to save output image (default: inpainted.png)'
    )
    parser.add_argument(
        '--detect',
        type=str,
        nargs='+',
        required=True,
        help='Object types to detect and remove (e.g., "person" "car" "sign")'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='a photorealistic plausible replacement',
        help='Prompt describing what to replace masked regions with'
    )
    parser.add_argument(
        '--negative_prompt',
        type=str,
        default=None,
        help='Negative prompt (what to avoid in generation)'
    )
    parser.add_argument(
        '--detection_threshold',
        type=float,
        default=0.3,
        help='Detection confidence threshold (default: 0.3)'
    )
    parser.add_argument(
        '--guidance_scale',
        type=float,
        default=8.0,
        help='Guidance scale for inpainting (default: 8.0)'
    )
    parser.add_argument(
        '--num_steps',
        type=int,
        default=20,
        help='Number of inference steps (default: 20)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--no_sam',
        action='store_true',
        help='Disable SAM (use bounding box masks)'
    )
    parser.add_argument(
        '--show_mask',
        action='store_true',
        help='Show the mask being used for inpainting'
    )
    
    return parser.parse_args()


def load_image(image_path: str) -> Image.Image:
    """Load image from file."""
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    print(f"✓ Image loaded successfully (size: {image.size})")
    return image


def visualize_results(
    original: Image.Image,
    inpainted: Image.Image,
    mask: torch.Tensor,
    output_path: str,
    detected_labels: list,
    show_mask: bool = True
):
    """
    Create and save visualization comparing original and inpainted images.
    
    Args:
        original: Original PIL image
        inpainted: Inpainted PIL image
        mask: Combined mask tensor [H, W]
        output_path: Path to save visualization
        detected_labels: List of detected object labels
        show_mask: Whether to show the mask
    """
    print(f"\nCreating visualization...")
    
    # Determine number of subplots
    n_plots = 3 if show_mask else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
    
    if n_plots == 2:
        axes = [axes[0], axes[1], None]
    
    # 1. Original image
    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Inpainted image
    axes[1].imshow(inpainted)
    axes[1].set_title('Inpainted Result', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Mask (optional)
    if show_mask and axes[2] is not None:
        mask_np = mask.cpu().numpy()
        axes[2].imshow(original)
        axes[2].imshow(mask_np, cmap='Reds', alpha=0.6)
        axes[2].set_title(f'Inpainted Regions\n({", ".join(set(detected_labels))})', 
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    
    plt.close()


def main():
    """Main function."""
    args = parse_args()
    
    # Validate paths
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print("="*60)
    print("INPAINTING WITH REGION DETECTION TEST")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Output: {args.output}")
    print(f"Detecting: {', '.join(args.detect)}")
    print(f"Prompt: {args.prompt}")
    if args.negative_prompt:
        print(f"Negative prompt: {args.negative_prompt}")
    print(f"Detection threshold: {args.detection_threshold}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Inference steps: {args.num_steps}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print(f"SAM: {'Disabled' if args.no_sam else 'Enabled'}")
    print("="*60)
    print()
    
    print(f"\nStep 4: Initializing SDXL inpainting...")
    inpainter = Inpainting(
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_steps
    )
    # Load image
    image = load_image(str(image_path))
    
    # Step 1: Initialize region extractor
    print("Step 1: Initializing region detection...")
    extractor = EditableRegionExtraction(
        text_labels=args.detect,
        threshold=args.detection_threshold,
        use_sam=not args.no_sam
    )
    print("✓ Region detector initialized")
    
    # Step 2: Detect regions
    print(f"\nStep 2: Detecting regions ({', '.join(args.detect)})...")
    with torch.no_grad():
        results = extractor(image, return_dict=True)
    
    masks = results['masks']
    labels = results['labels']
    scores = results['scores']
    
    if len(masks) == 0:
        print("\n⚠ No regions detected!")
        print("Try:")
        print("  - Lowering the detection threshold (--detection_threshold)")
        print("  - Using more general labels")
        return
    
    print(f"✓ Detected {len(masks)} region(s):")
    for i, (label, score) in enumerate(zip(labels, scores), 1):
        coverage = torch.sum(masks[i-1] > 0) / masks[i-1].numel() * 100
        print(f"  {i}. {label} (confidence: {score:.3f}, coverage: {coverage:.1f}%)")
    
    # Step 3: Combine masks
    print(f"\nStep 3: Combining masks...")
    combined_mask = torch.sum(masks, dim=0)
    combined_mask = torch.clamp(combined_mask, 0, 1)
    
    total_coverage = torch.sum(combined_mask > 0) / combined_mask.numel() * 100
    print(f"✓ Combined mask covers {total_coverage:.1f}% of image")
    
    # Step 4: Initialize inpainter
    print("✓ Inpainter initialized")
    
    # Step 5: Inpaint
    print(f"\nStep 5: Inpainting masked regions...")
    print(f"  Prompt: '{args.prompt}'")
    print(f"  (This may take 30-60 seconds...)")
    
    inpainted_image = inpainter(
        image=image,
        mask=combined_mask,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed
    )
    
    print("✓ Inpainting completed")
    
    # Step 6: Save results
    print(f"\nStep 6: Saving results...")
    
    # Save the inpainted image
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inpainted_image.save(output_path)
    print(f"✓ Inpainted image saved to: {output_path}")
    
    # Create visualization
    viz_path = output_path.parent / f"{output_path.stem}_comparison.png"
    visualize_results(
        image,
        inpainted_image,
        combined_mask,
        str(viz_path),
        labels,
        show_mask=args.show_mask
    )
    
    print("\n" + "="*60)
    print("✓ Test completed successfully!")
    print("="*60)
    print(f"\nResults:")
    print(f"  - Inpainted image: {output_path}")
    print(f"  - Comparison visualization: {viz_path}")


if __name__ == "__main__":
    main()
