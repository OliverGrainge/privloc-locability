# Geoprivacy Attribution Explanations

This module provides gradient-based attribution methods to identify which regions of an image contribute most to geolocation vulnerability.

## Overview

The `GeoAttributionHeatmap` class computes **∂(risk_score)/∂(image)** to identify spatial regions that most affect whether an image is localizable. This produces heatmaps showing which parts of an image leak location information.

## Quick Start

### Basic Usage

```python
from models import BinaryClassifier
from models.explanations import GeoAttributionHeatmap, save_attribution_visualization
from PIL import Image

# Load trained model
model = BinaryClassifier.load_from_checkpoint("path/to/checkpoint.ckpt")

# Create attribution system
attributor = GeoAttributionHeatmap(
    model=model,
    transform=model.transform,
    device="cuda"
)

# Load image
image = Image.open("test_image.jpg")

# Generate heatmap
heatmap, risk_score = attributor(image, return_risk_score=True)

# Save visualization
save_attribution_visualization(
    image=image,
    heatmap=heatmap[0],  # Remove batch dimension
    save_path="attribution.png",
    risk_score=risk_score[0].item()
)
```

### Command-Line Demo

```bash
# Basic usage
python demo_attribution.py \
    --checkpoint runs/binaryclassfier/megaloc/checkpoints/best_model.ckpt \
    --image test_image.jpg

# With custom options
python demo_attribution.py \
    --checkpoint path/to/model.ckpt \
    --image test_image.jpg \
    --output custom_output.png \
    --device cuda \
    --colormap hot \
    --alpha 0.6 \
    --method integrated
```

## Attribution Methods

### 1. Gradient × Input (Default)

Fast, single forward/backward pass:

```python
heatmap, risk_score = attributor(image)
```

- **Speed**: ~0.1s per image on GPU
- **Use case**: Quick exploration, batch processing
- **Formula**: `∂(risk)/∂(pixel) × pixel_value`

### 2. Integrated Gradients

More stable, averages gradients along path from black image to actual image:

```python
heatmap, risk_score = attributor.integrated_gradients(
    image, 
    n_steps=50
)
```

- **Speed**: ~5s per image on GPU (50 steps)
- **Use case**: Publication-quality figures, when you need more stable attributions
- **Formula**: `avg_gradients × (image - baseline)`

## API Reference

### `GeoAttributionHeatmap`

**Constructor:**

```python
GeoAttributionHeatmap(
    model,              # Trained BinaryClassifier
    transform,          # Image preprocessing transform
    device="cpu",       # Device to run on
    smooth_heatmap=True,  # Apply Gaussian smoothing
    smooth_kernel_size=15  # Smoothing kernel size
)
```

**Methods:**

- `__call__(image, return_risk_score=True)` - Generate gradient attribution
- `integrated_gradients(image, n_steps=50, return_risk_score=True)` - Generate integrated gradients attribution

**Returns:**
- If `return_risk_score=True`: `(heatmap [B, H, W], risk_scores [B])`
- If `return_risk_score=False`: `heatmap [B, H, W]`

### Visualization Functions

**`save_attribution_visualization`**
```python
save_attribution_visualization(
    image,           # Original image
    heatmap,         # Attribution map [H, W]
    save_path,       # Output file path
    risk_score=None, # Optional risk score
    colormap='jet',  # Heatmap colormap
    alpha=0.5,       # Overlay transparency
    dpi=150          # Output resolution
)
```

**`create_attribution_figure`**
```python
fig = create_attribution_figure(
    image, heatmap, risk_score=None, 
    title=None, colormap='jet', alpha=0.5
)
```
Returns matplotlib Figure with 3 panels: original, heatmap, overlay

**`overlay_heatmap_on_image`**
```python
overlaid_image = overlay_heatmap_on_image(
    image, heatmap, alpha=0.5, 
    colormap='jet', return_pil=True
)
```

**`highlight_top_regions`**
```python
highlighted = highlight_top_regions(
    image, heatmap, top_k_percent=10.0,
    highlight_color=(255, 0, 0), alpha=0.3
)
```

## Interpreting Results

### Risk Score
- **Range**: [0, 1]
- **Meaning**: Probability that geolocation error ≤ threshold_km
- **High score** (e.g., 0.8): Image is likely localizable within threshold
- **Low score** (e.g., 0.2): Image is likely NOT localizable within threshold

### Heatmap
- **Red/Hot regions**: High attribution - strongly affect localizability
- **Blue/Cool regions**: Low attribution - less important for localization
- **Normalized**: Values are relative within each image

### What Makes Regions "Risky"?
Common high-attribution regions:
- Distinctive landmarks (buildings, monuments)
- Text/signs (street signs, storefront names)
- Geographic features (mountains, coastlines)
- Architectural styles (region-specific)
- Vegetation patterns

## Advanced Usage

### Batch Processing

```python
# Process multiple images
images = [Image.open(f"img{i}.jpg") for i in range(10)]
heatmaps, risk_scores = attributor(images, return_risk_score=True)

# Save all
for i, (img, hmap, score) in enumerate(zip(images, heatmaps, risk_scores)):
    save_attribution_visualization(
        img, hmap, f"output_{i}.png", 
        risk_score=score.item()
    )
```

### Comparison Visualization

```python
from models.explanations import create_comparison_figure

fig = create_comparison_figure(
    images=[img1, img2, img3],
    heatmaps=[hmap1, hmap2, hmap3],
    risk_scores=[score1, score2, score3]
)
fig.savefig("comparison.png", dpi=150, bbox_inches='tight')
```

### Extract Top Regions

```python
from models.explanations import get_top_regions_mask, highlight_top_regions

# Get binary mask of top 10% regions
mask = get_top_regions_mask(heatmap, top_k_percent=10.0)

# Highlight them in red
highlighted = highlight_top_regions(
    image, heatmap, top_k_percent=10.0,
    highlight_color=(255, 0, 0)
)
highlighted.save("highlighted.png")
```

## Technical Details

### DINOv2 Architecture Specifics

The attribution method is optimized for Vision Transformer architectures:

- **Patch-based**: DINOv2 operates on 14×14 patches, but gradients backpropagate to pixels
- **Input size**: 518×518 (from your model's transform)
- **Output size**: Heatmap is same size as input (518×518)
- **Smoothing**: Applied to reduce patch boundary artifacts

### Memory Usage

- **Gradient method**: ~2GB GPU memory per image (518×518)
- **Integrated gradients**: ~2GB × n_steps (can be high)
- **Batch processing**: Linear scaling with batch size

### Performance

On NVIDIA A100:
- Gradient attribution: ~0.1s per image
- Integrated gradients (50 steps): ~5s per image

## Troubleshooting

**Issue**: Heatmap is too noisy
- **Solution**: Enable smoothing (default) or increase `smooth_kernel_size`

**Issue**: Heatmap shows patch grid patterns
- **Solution**: This is normal for ViT models; use smoothing to reduce

**Issue**: All regions have similar attribution
- **Solution**: Image may have uniform risk; check risk_score - if very low/high, the model is confident

**Issue**: Out of memory
- **Solution**: Reduce batch size or use smaller images

## Citation

If you use this attribution method in your research, please cite:

```bibtex
@inproceedings{yourpaper2025,
  title={Localized Geoprivacy Explanations: Identifying Image Regions that Contribute to Geolocation Vulnerability},
  author={Your Name},
  booktitle={Conference},
  year={2025}
}
```

## License

Same as parent project.
