#!/usr/bin/env python3
"""
Gradio Web App for Saliency-Based Region Inpainting.

This app provides an interactive interface for privacy-enhancing image editing
using geolocation saliency analysis, region detection, and AI inpainting.

Run with: python app.py
"""

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

from models.binaryclassifer import BinaryClassifier
from models.binaryclassifer.archs import load_arch_and_transform
from models.explanations.attribution import GeoAttributionHeatmap
from models.explanations.region import EditableRegionExtraction
from models.explanations.inpainting import Inpainting


# Global variables for caching models
_cached_geolocation_model = None
_cached_transform = None
_cached_attributor = None
_cached_extractor = None
_cached_inpainter = None


def load_geolocation_model(checkpoint_path: str, arch_name: str):
    """Load and cache geolocation model."""
    global _cached_geolocation_model, _cached_transform, _cached_attributor
    
    print(f"Loading geolocation model: {arch_name}")
    model = BinaryClassifier.load_from_checkpoint(
        checkpoint_path,
        arch_name=arch_name,
        strict=False
    )
    model.eval()
    
    _, transform = load_arch_and_transform(arch_name)
    attributor = GeoAttributionHeatmap(model, transform)
    
    _cached_geolocation_model = model
    _cached_transform = transform
    _cached_attributor = attributor
    
    return model, transform, attributor


def compute_region_saliency(masks: torch.Tensor, saliency: torch.Tensor) -> torch.Tensor:
    """Compute average saliency for each region."""
    # Resize saliency to match mask dimensions if needed
    if masks.shape[1:] != saliency.shape:
        import torch.nn.functional as F
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
        region_pixels = saliency_resized[mask > 0]
        if len(region_pixels) > 0:
            avg_saliency = region_pixels.mean().item()
        else:
            avg_saliency = 0.0
        scores.append(avg_saliency)
    
    return torch.tensor(scores)


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


def create_visualization(
    original: Image.Image,
    saliency: torch.Tensor,
    masks: torch.Tensor,
    labels: list,
    scores: torch.Tensor,
    region_saliencies: torch.Tensor,
    sorted_indices: list,
    selected_indices: list,
    inpainted: Image.Image,
    prob_before: float,
    prob_after: float
):
    """Create comprehensive visualization."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.2)
    
    # Row 1, Col 1: Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original)
    ax1.set_title(f'Original Image\nGeolocalizability: {prob_before:.3f}', 
                  fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Row 1, Col 2: Saliency heatmap
    ax2 = fig.add_subplot(gs[0, 1])
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
    
    # Row 1, Col 3: Detected regions
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(original)
    
    unique_labels = list(set(labels))
    colors = get_distinct_colors(len(unique_labels))
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    for i, (mask, label, score, sal) in enumerate(zip(masks, labels, scores, region_saliencies)):
        mask_np = mask.cpu().numpy()
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        if rows.any() and cols.any():
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            color = label_to_color[label]
            color_normalized = tuple(c / 255.0 for c in color)
            is_selected = i in selected_indices
            
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=4 if is_selected else 2,
                edgecolor='red' if is_selected else color_normalized,
                facecolor='none',
                linestyle='-' if is_selected else '--'
            )
            ax3.add_patch(rect)
            
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
    
    # Row 2, Col 1: Selected regions
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(original)
    
    height, width = masks[0].shape
    colored_overlay = np.zeros((height, width, 3))
    
    selected_colors = get_distinct_colors(len(selected_indices))
    for i, idx in enumerate(selected_indices):
        mask_np = masks[idx].cpu().numpy()
        color = np.array(selected_colors[i]) / 255.0
        
        for c in range(3):
            colored_overlay[:, :, c] += mask_np * color[c]
    
    colored_overlay = np.clip(colored_overlay, 0, 1)
    
    selected_mask = torch.zeros_like(masks[0])
    for idx in selected_indices:
        selected_mask += masks[idx]
    selected_mask = torch.clamp(selected_mask, 0, 1)
    alpha = selected_mask.cpu().numpy() * 0.7
    
    ax4.imshow(colored_overlay, alpha=alpha.max())
    
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
    ax5.set_title(f'Privacy-Enhanced Image\nGeolocalizability: {prob_after:.3f}', 
                  fontsize=14, fontweight='bold')
    ax5.axis('off')
    
    # Row 2, Col 3: Before/After comparison
    ax6 = fig.add_subplot(gs[1, 2])
    
    width_img, height_img = original.size
    comparison = Image.new('RGB', (width_img * 2, height_img))
    comparison.paste(original, (0, 0))
    comparison.paste(inpainted, (width_img, 0))
    
    ax6.imshow(comparison)
    ax6.axvline(x=width_img, color='white', linewidth=3, linestyle='--')
    ax6.text(width_img // 2, height_img - 30, 'BEFORE', 
             fontsize=16, fontweight='bold', color='white',
             ha='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
    ax6.text(width_img + width_img // 2, height_img - 30, 'AFTER', 
             fontsize=16, fontweight='bold', color='white',
             ha='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
    
    reduction = (prob_before - prob_after) / prob_before * 100
    ax6.set_title(f'Before vs After\nGeolocalizability Reduction: {reduction:.1f}%', 
                  fontsize=14, fontweight='bold')
    ax6.axis('off')
    
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    viz_image = Image.open(buf)
    plt.close()
    
    return viz_image


def process_image(
    image: Image.Image,
    checkpoint_path: str,
    arch_name: str,
    detect_labels: str,
    top_k: int,
    saliency_threshold: float,
    use_threshold: bool,
    prompt: str,
    negative_prompt: str,
    detection_threshold: float,
    use_sam: bool,
    seed: int,
    progress=gr.Progress()
):
    """Main processing function."""
    try:
        if image is None:
            return None, None, "Please upload an image first!"
        
        # Parse detection labels
        if detect_labels.strip():
            labels = [l.strip() for l in detect_labels.split(',')]
        else:
            labels = None
        
        progress(0.0, desc="Loading models...")
        
        # Load geolocation model
        model, transform, attributor = load_geolocation_model(checkpoint_path, arch_name)
        
        progress(0.1, desc="Computing saliency...")
        
        # Get initial prediction
        image_tensor = transform(image).unsqueeze(0).to(model.device)
        with torch.no_grad():
            logit_before = model(image_tensor)
            prob_before = torch.sigmoid(logit_before).item()
        
        # Compute saliency
        with torch.no_grad():
            saliency = attributor(image, head_fusion='mean', discard_ratio=0.1)
        
        saliency_min = saliency.min()
        saliency_max = saliency.max()
        saliency_normalized = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-8)
        
        progress(0.3, desc="Detecting regions...")
        
        # Extract regions
        global _cached_extractor
        if _cached_extractor is None:
            _cached_extractor = EditableRegionExtraction(
                text_labels=labels,
                threshold=detection_threshold,
                use_sam=use_sam
            )
        
        with torch.no_grad():
            regions_dict = _cached_extractor(image, return_dict=True)
        
        masks = regions_dict['masks']
        region_labels = regions_dict['labels']
        scores = regions_dict['scores']
        
        if len(masks) == 0:
            return None, None, "❌ No regions detected! Try lowering the detection threshold or using different labels."
        
        progress(0.5, desc="Ranking regions by saliency...")
        
        # Rank regions
        region_saliencies = compute_region_saliency(masks, saliency_normalized)
        sorted_indices = torch.argsort(region_saliencies, descending=True).tolist()
        
        # Select regions
        if use_threshold:
            selected_indices = [
                idx for idx in sorted_indices 
                if region_saliencies[idx].item() > saliency_threshold
            ]
        else:
            selected_indices = sorted_indices[:top_k]
        
        if len(selected_indices) == 0:
            return None, None, f"❌ No regions meet the criteria (threshold: {saliency_threshold})!"
        
        progress(0.6, desc=f"Combining {len(selected_indices)} region masks...")
        
        # Combine masks
        combined_mask = torch.zeros_like(masks[0])
        for idx in selected_indices:
            combined_mask += masks[idx]
        combined_mask = torch.clamp(combined_mask, 0, 1)
        
        progress(0.7, desc="Inpainting (this may take 30-60 seconds)...")
        
        # Inpaint
        global _cached_inpainter
        if _cached_inpainter is None:
            _cached_inpainter = Inpainting()
        
        inpainted_image = _cached_inpainter(
            image=image,
            mask=combined_mask,
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            seed=seed
        )
        
        progress(0.9, desc="Evaluating results...")
        
        # Evaluate
        inpainted_tensor = transform(inpainted_image).unsqueeze(0).to(model.device)
        with torch.no_grad():
            logit_after = model(inpainted_tensor)
            prob_after = torch.sigmoid(logit_after).item()
        
        reduction = (prob_before - prob_after) / prob_before * 100
        
        progress(0.95, desc="Creating visualization...")
        
        # Create visualization
        viz_image = create_visualization(
            image,
            saliency_normalized,
            masks,
            region_labels,
            scores,
            region_saliencies,
            sorted_indices,
            selected_indices,
            inpainted_image,
            prob_before,
            prob_after
        )
        
        # Create summary text
        summary = f"""
## ✅ Processing Complete!

### Privacy Metrics
- **Geolocalizability Before:** {prob_before:.4f}
- **Geolocalizability After:** {prob_after:.4f}
- **Reduction:** {reduction:.1f}%

### Regions Processed
- **Total Detected:** {len(masks)}
- **Regions Inpainted:** {len(selected_indices)}
- **Coverage Modified:** {(combined_mask > 0).sum().item() / combined_mask.numel() * 100:.1f}%

### Top Inpainted Regions
"""
        for i, idx in enumerate(selected_indices[:5], 1):
            label = region_labels[idx]
            sal = region_saliencies[idx].item()
            summary += f"{i}. **{label}** (saliency: {sal:.3f})\n"
        
        progress(1.0, desc="Done!")
        
        return viz_image, inpainted_image, summary
        
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Privacy-Enhancing Image Editor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔒 Privacy-Enhancing Image Editor
    
    This tool automatically identifies and removes the most geolocalizable parts of your images using:
    - **Saliency Analysis**: Identifies what makes images geolocalizable
    - **Region Detection**: Detects objects like buildings, signs, landmarks
    - **AI Inpainting**: Replaces sensitive regions with natural alternatives
    
    Upload an image and adjust the parameters to protect your privacy!
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input section
            gr.Markdown("### 📤 Input Image")
            input_image = gr.Image(type="pil", label="Upload Image")
            
            # Hidden/default values for model settings
            checkpoint_path = gr.Textbox(
                value="/home/oeg1n18/privloc-locability/runs/binaryclassfier/megaloc/checkpoints/best_model.ckpt",
                visible=False
            )
            arch_name = gr.Textbox(value="megaloc", visible=False)
            
            gr.Markdown("### 🎯 Detection Settings")
            detect_labels = gr.Textbox(
                value="building, sign, car, landmark, monument",
                label="Detection Labels (comma-separated)",
                info="Leave empty for defaults"
            )
            detection_threshold = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.3,
                step=0.05,
                label="Detection Confidence Threshold"
            )
            use_sam = gr.Checkbox(value=True, label="Use SAM for Precise Masks")
            
            gr.Markdown("### 🔝 Region Selection")
            use_threshold = gr.Checkbox(
                value=False,
                label="Use Saliency Threshold (instead of Top K)"
            )
            top_k = gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="Top K Regions to Inpaint",
                visible=True
            )
            saliency_threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.05,
                label="Saliency Threshold",
                visible=False
            )
            
            def toggle_selection_mode(use_thresh):
                return gr.update(visible=not use_thresh), gr.update(visible=use_thresh)
            
            use_threshold.change(
                toggle_selection_mode,
                inputs=[use_threshold],
                outputs=[top_k, saliency_threshold]
            )
            
            gr.Markdown("### 🎨 Inpainting Settings")
            prompt = gr.Textbox(
                value="a photorealistic natural replacement",
                label="Inpainting Prompt",
                info="Describe what to generate in place of removed regions"
            )
            negative_prompt = gr.Textbox(
                value="",
                label="Negative Prompt (Optional)",
                info="What to avoid in generation"
            )
            seed = gr.Slider(
                minimum=0,
                maximum=10000,
                value=42,
                step=1,
                label="Random Seed"
            )
            
            process_btn = gr.Button("🚀 Process Image", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            # Output section
            gr.Markdown("### 📊 Analysis & Results")
            output_viz = gr.Image(label="Comprehensive Analysis", type="pil")
            
            with gr.Row():
                output_image = gr.Image(label="Privacy-Enhanced Image", type="pil")
            
            output_text = gr.Markdown("Upload an image and click 'Process Image' to begin!")
    
    # Examples
    gr.Markdown("### 💡 Example Images")
    gr.Examples(
        examples=[
            ["assets/london.jpeg", 3, "natural urban scenery"],
            ["assets/paris.jpeg", 2, "generic architecture"],
            ["assets/rome.jpg", 3, "neutral background"],
        ],
        inputs=[input_image, top_k, prompt],
        label="Try these examples"
    )
    
    # Connect the button
    process_btn.click(
        fn=process_image,
        inputs=[
            input_image,
            checkpoint_path,
            arch_name,
            detect_labels,
            top_k,
            saliency_threshold,
            use_threshold,
            prompt,
            negative_prompt,
            detection_threshold,
            use_sam,
            seed
        ],
        outputs=[output_viz, output_image, output_text]
    )
    
    gr.Markdown("""
    ---
    ### 📖 How to Use
    
    1. **Upload an image** you want to make more private
    2. **Adjust detection labels** to target specific objects (buildings, signs, etc.)
    3. **Choose selection method**:
       - **Top K**: Remove the K most geolocalizable regions
       - **Threshold**: Remove all regions above a saliency threshold
    4. **Set inpainting prompt** to describe what should replace removed regions
    5. **Click "Process Image"** and wait 60-90 seconds
    6. **Download results**: Analysis visualization and privacy-enhanced image
    
    ### 🎯 Tips for Best Results
    
    - **For urban photos**: Use labels like "sign", "building", "landmark"
    - **For privacy**: Higher Top K = more aggressive removal
    - **For quality**: Enable SAM for precise masks (slower but better)
    - **For prompts**: Be specific about what to generate ("natural park scenery" vs "nature")
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
