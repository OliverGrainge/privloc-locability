#!/usr/bin/env python3
"""
Test script for editable region extraction.

This script uses Grounding DINO + SAM (Segment Anything Model) to detect 
and extract precise segmentation masks from input images based on text labels.

Usage:
    # With SAM for precise segmentation (default):
    python scripts/test_region_extraction.py \
        --image path/to/image.jpg \
        --output regions.png \
        --labels "building" "tree" "car" "person"
    
    # Without SAM (using bounding box masks):
    python scripts/test_region_extraction.py \
        --image path/to/image.jpg \
        --output regions.png \
        --no_sam
    
    # With different SAM model:
    python scripts/test_region_extraction.py \
        --image path/to/image.jpg \
        --sam_model_id facebook/sam-vit-huge
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from models.explanations.region import EditableRegionExtraction


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test editable region extraction on an image'
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
        default='regions.png',
        help='Path to save output visualization (default: regions.png)'
    )
    parser.add_argument(
        '--labels',
        type=str,
        nargs='+',
        default=None,
        help='Text labels to detect (space-separated). If not provided, uses defaults.'
    )
    parser.add_argument(
        '--model_id',
        type=str,
        default='IDEA-Research/grounding-dino-tiny',
        help='HuggingFace model ID (default: IDEA-Research/grounding-dino-tiny)'
    )
    parser.add_argument(
        '--sam_model_id',
        type=str,
        default='facebook/sam-vit-base',
        help='SAM model ID (default: facebook/sam-vit-base, options: sam-vit-huge, sam-vit-large, sam-vit-base)'
    )
    parser.add_argument(
        '--no_sam',
        action='store_true',
        help='Disable SAM and use rectangular bounding box masks instead'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.3,
        help='Detection confidence threshold (default: 0.3)'
    )
    parser.add_argument(
        '--text_threshold',
        type=float,
        default=0.25,
        help='Text matching threshold (default: 0.25)'
    )
    parser.add_argument(
        '--show_scores',
        action='store_true',
        help='Display confidence scores on masks'
    )
    
    return parser.parse_args()


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


def get_distinct_colors(n: int) -> list:
    """
    Generate n visually distinct colors.
    
    Args:
        n: Number of colors to generate
        
    Returns:
        List of RGB tuples
    """
    colors = []
    for i in range(n):
        hue = i / n
        # Convert HSV to RGB (using simple approximation)
        c = hue * 6
        x = 1 - abs(c % 2 - 1)
        
        if c < 1:
            r, g, b = 1, x, 0
        elif c < 2:
            r, g, b = x, 1, 0
        elif c < 3:
            r, g, b = 0, 1, x
        elif c < 4:
            r, g, b = 0, x, 1
        elif c < 5:
            r, g, b = x, 0, 1
        else:
            r, g, b = 1, 0, x
            
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    
    return colors


def visualize_regions(
    image: Image.Image,
    results: dict,
    output_path: str,
    show_scores: bool = True,
    use_sam: bool = True
):
    """
    Create and save region visualization with masks.
    
    Args:
        image: Original PIL image
        results: Detection results from EditableRegionExtraction (with return_dict=True)
        output_path: Path to save visualization
        show_scores: Whether to display confidence scores
        use_sam: Whether SAM was used (for display purposes)
    """
    print(f"\nCreating visualization...")
    
    masks = results['masks']
    labels = results['labels']
    scores = results['scores']
    
    mask_type = "SAM Segmentation" if use_sam else "Bounding Box"
    
    if len(masks) == 0:
        print("⚠ No regions detected!")
        return
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.2)
    
    # 1. Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. All masks combined
    ax2 = fig.add_subplot(gs[0, 1])
    combined_mask = torch.sum(masks, dim=0).cpu().numpy()
    combined_mask = np.clip(combined_mask, 0, 1)  # Clip overlaps
    ax2.imshow(image)
    im2 = ax2.imshow(combined_mask, cmap='jet', alpha=0.6, vmin=0, vmax=1)
    ax2.set_title(f'All Regions Combined ({len(masks)} masks)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    # Add colorbar
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 3. Individual colored masks overlay
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(image)
    ax3.set_title(f'Colored Mask Overlay ({mask_type})', fontsize=14, fontweight='bold')
    
    # Get unique labels and assign colors
    unique_labels = list(set(labels))
    colors = get_distinct_colors(len(unique_labels))
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # Create colored mask overlay
    height, width = masks.shape[1], masks.shape[2]
    colored_mask = np.zeros((height, width, 4))
    
    for i, (mask, label, score) in enumerate(zip(masks, labels, scores)):
        mask_np = mask.cpu().numpy()
        color = label_to_color[label]
        color_normalized = np.array([c / 255.0 for c in color] + [0.6])
        
        # Add colored mask
        for c in range(4):
            colored_mask[:, :, c] += mask_np * color_normalized[c]
    
    # Clip values to [0, 1]
    colored_mask = np.clip(colored_mask, 0, 1)
    ax3.imshow(colored_mask)
    ax3.axis('off')
    
    # Create legend
    legend_elements = [
        patches.Patch(
            facecolor=tuple(c / 255.0 for c in label_to_color[label]),
            edgecolor='black',
            label=label
        )
        for label in unique_labels
    ]
    ax3.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=9,
        framealpha=0.9
    )
    
    # 4. Individual masks grid (show up to 6 masks)
    num_masks_to_show = min(6, len(masks))
    for idx in range(num_masks_to_show):
        ax = fig.add_subplot(gs[1, idx % 3] if num_masks_to_show <= 3 else gs[1, idx % 3])
        
        mask_np = masks[idx].cpu().numpy()
        label = labels[idx]
        score = scores[idx].item()
        
        # Show mask overlay on image
        ax.imshow(image)
        ax.imshow(mask_np, cmap='Reds', alpha=0.6)
        
        if show_scores:
            title = f'{label} ({score:.2f})'
        else:
            title = label
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # If we have empty subplots, hide them
    if num_masks_to_show < 3:
        for idx in range(num_masks_to_show, 3):
            ax = fig.add_subplot(gs[1, idx])
            ax.axis('off')
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    
    plt.close()


def print_detection_summary(results: dict):
    """
    Print summary of detected regions.
    
    Args:
        results: Detection results from EditableRegionExtraction (with return_dict=True)
    """
    masks = results['masks']
    labels = results['labels']
    scores = results['scores']
    image_size = results['image_size']
    
    print(f"\nDetection Summary:")
    print(f"  Image size: {image_size}")
    print(f"  Total masks: {len(masks)}")
    
    if len(masks) == 0:
        return
    
    # Compute mask statistics
    masks_np = masks.cpu().numpy()
    scores_np = scores.cpu().numpy()
    
    print(f"  Mask shape: {masks.shape}")
    print(f"  Total pixels covered: {int(np.sum(masks_np > 0))}")
    
    # Group by label
    label_counts = {}
    label_scores = {}
    label_coverage = {}
    
    for i, label in enumerate(labels):
        if label not in label_counts:
            label_counts[label] = 0
            label_scores[label] = []
            label_coverage[label] = []
        
        label_counts[label] += 1
        label_scores[label].append(scores_np[i])
        # Calculate pixel coverage for this mask
        coverage = np.sum(masks_np[i] > 0) / (masks.shape[1] * masks.shape[2]) * 100
        label_coverage[label].append(coverage)
    
    print(f"\nDetections by label:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        avg_score = np.mean(label_scores[label])
        avg_coverage = np.mean(label_coverage[label])
        print(f"  {label}: {count} mask(s), avg confidence: {avg_score:.3f}, avg coverage: {avg_coverage:.1f}%")
    
    print(f"\nTop 10 masks by confidence:")
    sorted_indices = np.argsort(scores_np)[::-1][:10]
    for i, idx in enumerate(sorted_indices, 1):
        label = labels[idx]
        score = scores_np[idx]
        coverage = np.sum(masks_np[idx] > 0) / (masks.shape[1] * masks.shape[2]) * 100
        print(f"  {i}. {label} (confidence: {score:.3f}, coverage: {coverage:.1f}%)")


def main():
    """Main function."""
    args = parse_args()
    
    # Validate paths
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print("="*60)
    print("EDITABLE REGION EXTRACTION TEST")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Output: {args.output}")
    print(f"Grounding DINO Model: {args.model_id}")
    if not args.no_sam:
        print(f"SAM Model: {args.sam_model_id}")
    else:
        print(f"SAM: Disabled (using bounding box masks)")
    print(f"Detection threshold: {args.threshold}")
    print(f"Text threshold: {args.text_threshold}")
    if args.labels:
        print(f"Labels: {', '.join(args.labels)}")
    else:
        print(f"Labels: Using defaults")
    print("="*60)
    print()
    
    # Initialize region extractor
    print("Initializing EditableRegionExtraction...")
    print("(This may take a moment to download the models...)")
    
    extractor = EditableRegionExtraction(
        text_labels=args.labels,
        model_id=args.model_id,
        sam_model_id=args.sam_model_id,
        threshold=args.threshold,
        text_threshold=args.text_threshold,
        use_sam=not args.no_sam
    )
    print("✓ Region extractor initialized")
    
    if args.labels is None:
        print(f"✓ Using default labels: {', '.join(extractor.text_labels)}")
    
    # Load image
    image = load_image(str(image_path))
    
    # Extract regions
    print(f"\nExtracting regions as masks...")
    print(f"  (This may take a moment...)")
    
    with torch.no_grad():
        results = extractor(image, return_dict=True)
    
    print("✓ Region extraction completed successfully")
    
    # Print detection summary
    print_detection_summary(results)
    
    # Create visualization
    if len(results['masks']) > 0:
        visualize_regions(
            image,
            results,
            args.output,
            show_scores=args.show_scores,
            use_sam=not args.no_sam
        )
    else:
        print("\n⚠ Skipping visualization (no regions detected)")
        print("  Try:")
        print("  - Lowering the detection threshold (--threshold)")
        print("  - Lowering the text threshold (--text_threshold)")
        print("  - Using different or more general labels (--labels)")
    
    print("\n" + "="*60)
    print("✓ Test completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
