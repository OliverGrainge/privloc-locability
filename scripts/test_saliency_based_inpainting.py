#!/usr/bin/env python3
"""
Test script for saliency-based region inpainting.

This script combines attribution (saliency), region extraction, and inpainting
to identify and remove the most geolocalizable parts of an image.

Workflow:
    1. Compute saliency map showing which parts are important for geolocation
    2. Extract editable regions (objects) from the image
    3. Rank regions by their average saliency (geolocation importance)
    4. Inpaint the most important regions to reduce geolocalizability

Usage:
    python scripts/test_saliency_based_inpainting.py \
        --image path/to/image.jpg \
        --checkpoint path/to/model.ckpt \
        --arch_name megaloc \
        --detect "building" "sign" "car" \
        --top_k 3 \
        --prompt "natural scenery"
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from models.binaryclassifer import BinaryClassifier
from models.binaryclassifer.archs import load_arch_and_transform
from models.explanations.attribution import GeoAttributionHeatmap
from models.explanations.region import EditableRegionExtraction
from models.explanations.inpainting import Inpainting


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Saliency-based region inpainting for privacy protection'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/home/oeg1n18/privloc-locability/runs/binaryclassfier/megaloc/checkpoints/best_model.ckpt',
        help='Path to geolocation model checkpoint (.ckpt file)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='saliency_inpainted.png',
        help='Path to save final inpainted image (default: saliency_inpainted.png)'
    )
    parser.add_argument(
        '--arch_name',
        type=str,
        default='megaloc',
        help='Architecture name (default: megaloc)'
    )
    parser.add_argument(
        '--detect',
        type=str,
        nargs='+',
        default=None,
        help='Object types to detect (if None, uses defaults)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=None,
        help='Number of top salient regions to inpaint (if None, uses threshold)'
    )
    parser.add_argument(
        '--saliency_threshold',
        type=float,
        default=0.5,
        help='Only inpaint regions with avg saliency above this (default: 0.5, ignored if top_k is set)'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='a photorealistic natural replacement',
        help='Prompt for inpainting'
    )
    parser.add_argument(
        '--detection_threshold',
        type=float,
        default=0.3,
        help='Detection confidence threshold (default: 0.3)'
    )
    parser.add_argument(
        '--no_sam',
        action='store_true',
        help='Disable SAM (use bounding box masks)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for inpainting (default: 42)'
    )
    
    return parser.parse_args()


def load_model(checkpoint_path: str, arch_name: str):
    """Load geolocation model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    model = BinaryClassifier.load_from_checkpoint(
        checkpoint_path,
        arch_name=arch_name,
        strict=False
    )
    model.eval()
    print("✓ Model loaded successfully")
    return model


def load_image(image_path: str) -> Image.Image:
    """Load image from file."""
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    print(f"✓ Image loaded (size: {image.size})")
    return image


def compute_region_saliency(masks: torch.Tensor, saliency: torch.Tensor) -> torch.Tensor:
    """
    Compute average saliency for each region.
    
    Args:
        masks: Region masks [N, H, W]
        saliency: Saliency map [H_sal, W_sal] (may be different size)
        
    Returns:
        Average saliency per region [N]
    """
    # Resize saliency to match mask dimensions if needed
    if masks.shape[1:] != saliency.shape:
        import torch.nn.functional as F
        # Add batch and channel dimensions for interpolation
        saliency_resized = F.interpolate(
            saliency.unsqueeze(0).unsqueeze(0),
            size=masks.shape[1:],
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)
    else:
        saliency_resized = saliency
    
    scores = []
    for mask in masks:
        # Get pixels in this region
        region_pixels = saliency_resized[mask > 0]
        if len(region_pixels) > 0:
            avg_saliency = region_pixels.mean().item()
        else:
            avg_saliency = 0.0
        scores.append(avg_saliency)
    
    return torch.tensor(scores)


def visualize_comprehensive_results(
    original: Image.Image,
    saliency: torch.Tensor,
    regions_dict: dict,
    region_saliencies: torch.Tensor,
    sorted_indices: list,
    selected_indices: list,
    inpainted: Image.Image,
    output_path: str,
    model_prediction_before: float,
    model_prediction_after: float
):
    """
    Create comprehensive visualization of the entire pipeline.
    
    Shows: original, saliency, detected regions, ranked regions, and final result.
    """
    print("\nCreating comprehensive visualization...")
    
    masks = regions_dict['masks']
    labels = regions_dict['labels']
    scores = regions_dict['scores']
    
    # Create figure with 2 rows, 3 columns
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.2)
    
    # Row 1, Col 1: Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original)
    ax1.set_title(f'Original Image\nGeolocalizability: {model_prediction_before:.3f}', 
                  fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Row 1, Col 2: Saliency heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Resize saliency to match image dimensions for visualization
    original_array = np.array(original)
    target_height, target_width = original_array.shape[:2]
    
    if saliency.shape != (target_height, target_width):
        import torch.nn.functional as F
        saliency_resized = F.interpolate(
            saliency.unsqueeze(0).unsqueeze(0),
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)
        saliency_np = saliency_resized.cpu().numpy()
    else:
        saliency_np = saliency.cpu().numpy()
    
    ax2.imshow(original)
    im = ax2.imshow(saliency_np, cmap='jet', alpha=0.6)
    ax2.set_title('Geolocation Saliency Map\n(Red = High importance)', 
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # Row 1, Col 3: Detected regions with saliency scores
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(original)
    
    # Get unique labels for coloring
    unique_labels = list(set(labels))
    colors = get_distinct_colors(len(unique_labels))
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # Draw all detected regions
    for i, (mask, label, score, sal) in enumerate(zip(masks, labels, scores, region_saliencies)):
        # Get bounding box from mask
        mask_np = mask.cpu().numpy()
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        if rows.any() and cols.any():
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            color = label_to_color[label]
            color_normalized = tuple(c / 255.0 for c in color)
            
            is_selected = i in selected_indices
            
            # Draw bounding box (thicker and solid for selected regions)
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=4 if is_selected else 2,
                edgecolor='red' if is_selected else color_normalized,
                facecolor='none',
                linestyle='-' if is_selected else '--'
            )
            ax3.add_patch(rect)
            
            # Add label with saliency score
            rank = sorted_indices.index(i) + 1
            text = f"#{rank} {label}\nSal: {sal:.2f}"
            
            ax3.text(
                xmin, ymin - 5,
                text,
                color='white',
                fontsize=10,
                fontweight='bold' if is_selected else 'normal',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='red' if is_selected else color_normalized,
                    edgecolor='yellow' if is_selected else 'none',
                    linewidth=3 if is_selected else 0,
                    alpha=0.95 if is_selected else 0.8
                )
            )
    
    ax3.set_title(f'Detected Regions (N={len(masks)})\nRanked by Saliency', 
                  fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Row 2, Col 1: Top salient regions (to be inpainted)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(original)
    
    # Create a colored overlay showing each selected region distinctly
    height, width = masks[0].shape
    colored_overlay = np.zeros((height, width, 3))
    
    # Assign different colors to each selected region
    selected_colors = get_distinct_colors(len(selected_indices))
    for i, idx in enumerate(selected_indices):
        mask_np = masks[idx].cpu().numpy()
        color = np.array(selected_colors[i]) / 255.0
        
        # Add this region's color to the overlay
        for c in range(3):
            colored_overlay[:, :, c] += mask_np * color[c]
    
    # Normalize and clip
    colored_overlay = np.clip(colored_overlay, 0, 1)
    
    # Create alpha channel from combined mask
    selected_mask = torch.zeros_like(masks[0])
    for idx in selected_indices:
        selected_mask += masks[idx]
    selected_mask = torch.clamp(selected_mask, 0, 1)
    alpha = selected_mask.cpu().numpy() * 0.7
    
    # Blend with original image
    ax4.imshow(colored_overlay, alpha=alpha.max())
    
    # Draw bounding boxes around each selected region
    for i, idx in enumerate(selected_indices):
        mask_np = masks[idx].cpu().numpy()
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        if rows.any() and cols.any():
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            color = np.array(selected_colors[i]) / 255.0
            
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=3,
                edgecolor=color,
                facecolor='none'
            )
            ax4.add_patch(rect)
            
            # Add region number
            ax4.text(
                xmin, ymin - 5,
                f"#{i+1}: {labels[idx]}\n{region_saliencies[idx]:.3f}",
                color='white',
                fontsize=10,
                fontweight='bold',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor=color,
                    edgecolor='white',
                    linewidth=2,
                    alpha=0.9
                )
            )
    
    ax4.set_title(f'Top {len(selected_indices)} Most Geolocalizable Regions\n(To be inpainted)', 
                  fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    # Row 2, Col 2: Inpainted result
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(inpainted)
    ax5.set_title(f'Privacy-Enhanced Image\nGeolocalizability: {model_prediction_after:.3f}', 
                  fontsize=14, fontweight='bold')
    ax5.axis('off')
    
    # Row 2, Col 3: Before/After comparison
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Create side-by-side comparison
    width, height = original.size
    comparison = Image.new('RGB', (width * 2, height))
    comparison.paste(original, (0, 0))
    comparison.paste(inpainted, (width, 0))
    
    ax6.imshow(comparison)
    ax6.axvline(x=width, color='white', linewidth=3, linestyle='--')
    ax6.text(width // 2, height - 30, 'BEFORE', 
             fontsize=16, fontweight='bold', color='white',
             ha='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
    ax6.text(width + width // 2, height - 30, 'AFTER', 
             fontsize=16, fontweight='bold', color='white',
             ha='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
    
    reduction = (model_prediction_before - model_prediction_after) / model_prediction_before * 100
    ax6.set_title(f'Before vs After\nGeolocalizability Reduction: {reduction:.1f}%', 
                  fontsize=14, fontweight='bold')
    ax6.axis('off')
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    viz_path = output_path.parent / f"{output_path.stem}_analysis.png"
    plt.savefig(str(viz_path), dpi=150, bbox_inches='tight')
    print(f"✓ Analysis visualization saved to: {viz_path}")
    
    plt.close()


def get_distinct_colors(n: int) -> list:
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        hue = i / n
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


def main():
    """Main function."""
    args = parse_args()
    
    # Validate paths
    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print("="*70)
    print("SALIENCY-BASED REGION INPAINTING FOR PRIVACY PROTECTION")
    print("="*70)
    print(f"Image: {image_path}")
    print(f"Model: {checkpoint_path}")
    print(f"Architecture: {args.arch_name}")
    print(f"Output: {args.output}")
    if args.detect:
        print(f"Detecting: {', '.join(args.detect)}")
    print(f"Selection: Top {args.top_k} regions" if args.top_k else f"Saliency threshold > {args.saliency_threshold}")
    print(f"Prompt: {args.prompt}")
    print("="*70)
    print()
    
    # Load image
    image = load_image(str(image_path))
    
    # Step 1: Load geolocation model
    print("Step 1: Loading geolocation model...")
    model = load_model(str(checkpoint_path), args.arch_name)
    _, transform = load_arch_and_transform(args.arch_name)
    
    # Get initial prediction
    image_tensor = transform(image).unsqueeze(0).to(model.device)
    with torch.no_grad():
        logit_before = model(image_tensor)
        prob_before = torch.sigmoid(logit_before).item()
    print(f"✓ Initial geolocalizability score: {prob_before:.4f}")
    
    # Step 2: Compute saliency map
    print("\nStep 2: Computing saliency map...")
    attributor = GeoAttributionHeatmap(model, transform)
    with torch.no_grad():
        saliency = attributor(image, head_fusion='mean', discard_ratio=0.1)
    
    # Normalize saliency to [0, 1]
    saliency_min = saliency.min()
    saliency_max = saliency.max()
    saliency_normalized = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-8)
    
    print(f"✓ Saliency computed (range: {saliency_min:.4f} to {saliency_max:.4f})")
    
    # Step 3: Extract regions
    print("\nStep 3: Extracting editable regions...")
    extractor = EditableRegionExtraction(
        text_labels=args.detect,
        threshold=args.detection_threshold,
        use_sam=not args.no_sam
    )
    
    with torch.no_grad():
        regions_dict = extractor(image, return_dict=True)
    
    masks = regions_dict['masks']
    labels = regions_dict['labels']
    scores = regions_dict['scores']
    
    if len(masks) == 0:
        print("\n⚠ No regions detected!")
        print("Try lowering --detection_threshold or using different labels")
        return
    
    print(f"✓ Detected {len(masks)} regions")
    
    # Step 4: Rank regions by saliency
    print("\nStep 4: Ranking regions by geolocation importance...")
    region_saliencies = compute_region_saliency(masks, saliency_normalized)
    
    # Sort by saliency (descending)
    sorted_indices = torch.argsort(region_saliencies, descending=True).tolist()
    
    print(f"\nRegion saliency scores:")
    for rank, idx in enumerate(sorted_indices, 1):
        label = labels[idx]
        sal = region_saliencies[idx].item()
        conf = scores[idx].item()
        coverage = (masks[idx] > 0).sum().item() / masks[idx].numel() * 100
        print(f"  {rank}. {label:12s} | Saliency: {sal:.3f} | Confidence: {conf:.3f} | Coverage: {coverage:.1f}%")
    
    # Step 5: Select regions to inpaint
    print("\nStep 5: Selecting regions to inpaint...")
    if args.top_k is not None:
        # Select top K most salient regions
        selected_indices = sorted_indices[:args.top_k]
        print(f"✓ Selected top {len(selected_indices)} most salient regions")
    else:
        # Select regions above saliency threshold
        selected_indices = [
            idx for idx in sorted_indices 
            if region_saliencies[idx].item() > args.saliency_threshold
        ]
        print(f"✓ Selected {len(selected_indices)} regions with saliency > {args.saliency_threshold}")
    
    if len(selected_indices) == 0:
        print("\n⚠ No regions meet the criteria!")
        print(f"Try lowering --saliency_threshold (currently {args.saliency_threshold})")
        return
    
    print(f"\nRegions to be inpainted:")
    for idx in selected_indices:
        print(f"  - {labels[idx]} (saliency: {region_saliencies[idx]:.3f})")
    
    # Step 6: Combine selected masks
    print("\nStep 6: Combining selected region masks...")
    combined_mask = torch.zeros_like(masks[0])
    for idx in selected_indices:
        combined_mask += masks[idx]
    combined_mask = torch.clamp(combined_mask, 0, 1)
    
    coverage = (combined_mask > 0).sum().item() / combined_mask.numel() * 100
    print(f"✓ Combined mask covers {coverage:.1f}% of image")
    
    # Step 7: Inpaint
    print("\nStep 7: Inpainting selected regions...")
    print(f"  Prompt: '{args.prompt}'")
    print(f"  (This may take 30-60 seconds...)")
    
    inpainter = Inpainting()
    inpainted_image = inpainter(
        image=image,
        mask=combined_mask,
        prompt=args.prompt,
        seed=args.seed
    )
    
    print("✓ Inpainting completed")
    
    # Step 8: Evaluate privacy improvement
    print("\nStep 8: Evaluating privacy improvement...")
    inpainted_tensor = transform(inpainted_image).unsqueeze(0).to(model.device)
    with torch.no_grad():
        logit_after = model(inpainted_tensor)
        prob_after = torch.sigmoid(logit_after).item()
    
    reduction = (prob_before - prob_after) / prob_before * 100
    print(f"✓ Geolocalizability: {prob_before:.4f} → {prob_after:.4f}")
    print(f"✓ Reduction: {reduction:.1f}%")
    
    # Step 9: Save results
    print("\nStep 9: Saving results...")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inpainted_image.save(output_path)
    print(f"✓ Inpainted image saved to: {output_path}")
    
    # Create comprehensive visualization
    visualize_comprehensive_results(
        image,
        saliency_normalized,
        regions_dict,
        region_saliencies,
        sorted_indices,
        selected_indices,
        inpainted_image,
        str(output_path),
        prob_before,
        prob_after
    )
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nResults:")
    print(f"  - Privacy-enhanced image: {output_path}")
    print(f"  - Analysis visualization: {output_path.parent / f'{output_path.stem}_analysis.png'}")
    print(f"\nPrivacy Metrics:")
    print(f"  - Geolocalizability reduction: {reduction:.1f}%")
    print(f"  - Image coverage modified: {coverage:.1f}%")
    print(f"  - Regions inpainted: {len(selected_indices)}")


if __name__ == "__main__":
    main()
