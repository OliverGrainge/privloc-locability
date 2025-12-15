# Explainable AI Models

This directory contains explainability methods for geolocation models.

## Editable Region Extraction

The `EditableRegionExtraction` class combines **Grounding DINO** (for object detection) with **SAM** (Segment Anything Model) to generate precise segmentation masks based on text labels.

### Features

- **Text-based detection**: Detect objects using natural language descriptions
- **Precise segmentation**: Uses SAM to generate pixel-accurate masks (not just bounding boxes)
- **Flexible labels**: Supports custom text labels or uses sensible defaults
- **GPU accelerated**: Automatically uses available GPU via Accelerator

### Usage

#### Basic Usage

```python
from models.explanations.region import EditableRegionExtraction
from PIL import Image

# Initialize the model
extractor = EditableRegionExtraction(
    text_labels=["building", "tree", "car", "person"],
    threshold=0.3,
    use_sam=True  # Use SAM for precise segmentation
)

# Load an image
image = Image.open("path/to/image.jpg")

# Extract masks (returns tensor of shape [N, H, W])
masks = extractor(image)

# Or get detailed results with labels and scores
results = extractor(image, return_dict=True)
# results = {
#     'masks': torch.Tensor,      # [N, H, W]
#     'labels': List[str],        # ['building', 'tree', ...]
#     'scores': torch.Tensor,     # [N] confidence scores
#     'image_size': Tuple[int]    # (width, height)
# }
```

#### Without SAM (Bounding Box Masks)

If you want faster inference or don't need precise segmentation:

```python
extractor = EditableRegionExtraction(
    text_labels=["building", "tree"],
    use_sam=False  # Use rectangular bounding box masks
)

masks = extractor(image)
```

#### Custom Models

You can specify different model variants:

```python
extractor = EditableRegionExtraction(
    model_id="IDEA-Research/grounding-dino-base",  # or -tiny, -large
    sam_model_id="facebook/sam-vit-huge",          # or -base, -large
    threshold=0.4,
    text_threshold=0.3
)
```

### Command-Line Testing

Test the region extraction on any image:

```bash
# With SAM (precise segmentation)
python scripts/test_region_extraction.py \
    --image path/to/image.jpg \
    --output results.png \
    --labels "building" "tree" "car" "person"

# Without SAM (faster, rectangular masks)
python scripts/test_region_extraction.py \
    --image path/to/image.jpg \
    --output results.png \
    --no_sam

# Adjust detection thresholds
python scripts/test_region_extraction.py \
    --image path/to/image.jpg \
    --threshold 0.2 \
    --text_threshold 0.2

# Use larger SAM model for better quality
python scripts/test_region_extraction.py \
    --image path/to/image.jpg \
    --sam_model_id facebook/sam-vit-huge
```

### Default Labels

If no custom labels are provided, the following defaults are used:

- building
- tree
- car
- person
- animal
- plant
- object
- sky
- water

### Parameters

#### `__init__` Parameters

- `text_labels` (Optional[List[str]]): Text descriptions of objects to detect. If None, uses defaults.
- `model_id` (str): HuggingFace model ID for Grounding DINO. Default: `"IDEA-Research/grounding-dino-tiny"`
- `sam_model_id` (str): HuggingFace model ID for SAM. Default: `"facebook/sam-vit-base"`
- `threshold` (float): Detection confidence threshold. Default: `0.4`
- `text_threshold` (float): Text matching threshold. Default: `0.3`
- `use_sam` (bool): Whether to use SAM for precise segmentation. Default: `True`

#### `__call__` Parameters

- `image` (Union[torch.Tensor, Image.Image, List[Image.Image]]): Input image(s)
- `return_dict` (bool): If True, returns dictionary with masks and metadata. If False, returns only masks tensor. Default: `False`

### Model Comparison

| Feature | Without SAM | With SAM |
|---------|-------------|----------|
| Mask Type | Rectangular boxes | Precise object contours |
| Speed | Fast (~100ms) | Slower (~500ms) |
| Accuracy | Approximate | High precision |
| GPU Memory | Low | Higher |
| Use Case | Quick prototyping | Production, analysis |

### Performance Tips

1. **Use smaller models for speed**: `grounding-dino-tiny` + `sam-vit-base`
2. **Use larger models for quality**: `grounding-dino-base` + `sam-vit-huge`
3. **Disable SAM for quick testing**: Set `use_sam=False`
4. **Adjust thresholds**: Lower thresholds detect more (but with more false positives)
5. **Batch processing**: Currently processes one image at a time

### How It Works

1. **Grounding DINO** detects objects based on text labels and returns bounding boxes
2. **SAM** (if enabled) takes each bounding box as a prompt and generates precise segmentation masks
3. The output is a tensor of binary masks where each mask corresponds to one detected region

This approach combines the flexibility of text-based detection with the precision of state-of-the-art segmentation.

## Attribution Methods

See `attribution.py` for attention-based attribution methods like Attention Rollout.
