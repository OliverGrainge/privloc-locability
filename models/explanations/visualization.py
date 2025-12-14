"""
Visualization utilities for geoprivacy attribution heatmaps.

Provides functions to overlay heatmaps on images and create publication-quality figures.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from typing import Union, Optional, Tuple
from PIL import Image
import torchvision.transforms.functional as TF


def overlay_heatmap_on_image(
    image: Union[torch.Tensor, np.ndarray, Image.Image],
    heatmap: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.5,
    colormap: str = 'jet',
    return_pil: bool = True
) -> Union[Image.Image, np.ndarray]:
    """
    Overlay attribution heatmap on original image.
    
    Args:
        image: Original image [H, W, 3] or [3, H, W] or PIL Image
        heatmap: Attribution heatmap [H, W] with values in [0, 1]
        alpha: Transparency of heatmap overlay (0=invisible, 1=opaque)
        colormap: Matplotlib colormap name ('jet', 'hot', 'viridis', etc.)
        return_pil: If True, return PIL Image; if False, return numpy array
        
    Returns:
        Overlaid image as PIL Image or numpy array [H, W, 3]
    """
    # Convert to numpy arrays
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
        image = image.cpu().numpy()
    elif isinstance(image, Image.Image):
        image = np.array(image)
    
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    
    # Ensure image is in [0, 1] range
    if image.max() > 1.0:
        image = image / 255.0
    
    # Ensure heatmap is 2D
    if heatmap.ndim != 2:
        raise ValueError(f"Heatmap must be 2D, got shape {heatmap.shape}")
    
    # Resize heatmap to match image size if needed
    if heatmap.shape != image.shape[:2]:
        from scipy.ndimage import zoom
        zoom_factors = (image.shape[0] / heatmap.shape[0], 
                       image.shape[1] / heatmap.shape[1])
        heatmap = zoom(heatmap, zoom_factors, order=1)
    
    # Apply colormap to heatmap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[..., :3]  # [H, W, 3] RGB
    
    # Blend image and heatmap
    overlaid = (1 - alpha) * image + alpha * heatmap_colored
    overlaid = np.clip(overlaid, 0, 1)
    
    # Convert to uint8
    overlaid = (overlaid * 255).astype(np.uint8)
    
    if return_pil:
        return Image.fromarray(overlaid)
    else:
        return overlaid


def create_attribution_figure(
    image: Union[torch.Tensor, np.ndarray, Image.Image],
    heatmap: Union[torch.Tensor, np.ndarray],
    risk_score: Optional[float] = None,
    title: Optional[str] = None,
    colormap: str = 'jet',
    alpha: float = 0.5,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Create a comprehensive figure showing original image, heatmap, and overlay.
    
    Args:
        image: Original image
        heatmap: Attribution heatmap [H, W]
        risk_score: Optional risk score to display
        title: Optional title for the figure
        colormap: Matplotlib colormap name
        alpha: Transparency for overlay
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object
    """
    # Convert to numpy
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()
    elif isinstance(image, Image.Image):
        image = np.array(image)
    
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    
    # Ensure image is in [0, 1]
    if image.max() > 1.0:
        image = image / 255.0
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Heatmap only
    im = axes[1].imshow(heatmap, cmap=colormap, vmin=0, vmax=1)
    axes[1].set_title('Attribution Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Attribution Score', rotation=270, labelpad=15)
    
    # 3. Overlay
    overlaid = overlay_heatmap_on_image(image, heatmap, alpha=alpha, 
                                       colormap=colormap, return_pil=False)
    axes[2].imshow(overlaid)
    axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Add main title with risk score if provided
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    elif risk_score is not None:
        fig.suptitle(f'Geoprivacy Risk Score: {risk_score:.4f}', 
                    fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig


def create_comparison_figure(
    images: list,
    heatmaps: list,
    risk_scores: Optional[list] = None,
    titles: Optional[list] = None,
    colormap: str = 'jet',
    alpha: float = 0.5,
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Create a comparison figure with multiple images and their heatmaps.
    
    Args:
        images: List of images
        heatmaps: List of attribution heatmaps
        risk_scores: Optional list of risk scores
        titles: Optional list of titles for each image
        colormap: Matplotlib colormap name
        alpha: Transparency for overlay
        figsize: Figure size (auto-computed if None)
        
    Returns:
        Matplotlib figure object
    """
    n_images = len(images)
    if figsize is None:
        figsize = (5 * n_images, 10)
    
    # Create figure: 3 rows (original, heatmap, overlay) × n_images columns
    fig, axes = plt.subplots(3, n_images, figsize=figsize)
    
    if n_images == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_images):
        image = images[i]
        heatmap = heatmaps[i]
        
        # Convert to numpy
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] == 3:
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.cpu().numpy()
        
        if image.max() > 1.0:
            image = image / 255.0
        
        # Row 1: Original image
        axes[0, i].imshow(image)
        if titles and i < len(titles):
            axes[0, i].set_title(titles[i], fontsize=10, fontweight='bold')
        elif risk_scores and i < len(risk_scores):
            axes[0, i].set_title(f'Risk: {risk_scores[i]:.4f}', fontsize=10)
        axes[0, i].axis('off')
        
        # Row 2: Heatmap
        im = axes[1, i].imshow(heatmap, cmap=colormap, vmin=0, vmax=1)
        axes[1, i].axis('off')
        
        # Row 3: Overlay
        overlaid = overlay_heatmap_on_image(image, heatmap, alpha=alpha,
                                           colormap=colormap, return_pil=False)
        axes[2, i].imshow(overlaid)
        axes[2, i].axis('off')
    
    # Add row labels
    axes[0, 0].set_ylabel('Original', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Heatmap', fontsize=12, fontweight='bold')
    axes[2, 0].set_ylabel('Overlay', fontsize=12, fontweight='bold')
    
    # Add a single colorbar for all heatmaps
    fig.colorbar(im, ax=axes[1, :], location='bottom', 
                 fraction=0.046, pad=0.04, label='Attribution Score')
    
    plt.tight_layout()
    return fig


def save_attribution_visualization(
    image: Union[torch.Tensor, np.ndarray, Image.Image],
    heatmap: Union[torch.Tensor, np.ndarray],
    save_path: str,
    risk_score: Optional[float] = None,
    title: Optional[str] = None,
    colormap: str = 'jet',
    alpha: float = 0.5,
    dpi: int = 150
):
    """
    Create and save attribution visualization to file.
    
    Args:
        image: Original image
        heatmap: Attribution heatmap
        save_path: Path to save figure
        risk_score: Optional risk score to display
        title: Optional title
        colormap: Matplotlib colormap name
        alpha: Transparency for overlay
        dpi: Resolution for saved figure
    """
    fig = create_attribution_figure(
        image=image,
        heatmap=heatmap,
        risk_score=risk_score,
        title=title,
        colormap=colormap,
        alpha=alpha
    )
    
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved attribution visualization to: {save_path}")


def get_top_regions_mask(
    heatmap: Union[torch.Tensor, np.ndarray],
    top_k_percent: float = 10.0
) -> np.ndarray:
    """
    Extract binary mask of top-k% most important regions.
    
    Args:
        heatmap: Attribution heatmap [H, W]
        top_k_percent: Percentage of pixels to mark (e.g., 10.0 = top 10%)
        
    Returns:
        Binary mask [H, W] where 1 = top-k region, 0 = elsewhere
    """
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    
    # Flatten and get threshold
    flat = heatmap.flatten()
    threshold = np.percentile(flat, 100 - top_k_percent)
    
    # Create binary mask
    mask = (heatmap >= threshold).astype(np.uint8)
    return mask


def highlight_top_regions(
    image: Union[torch.Tensor, np.ndarray, Image.Image],
    heatmap: Union[torch.Tensor, np.ndarray],
    top_k_percent: float = 10.0,
    highlight_color: Tuple[int, int, int] = (255, 0, 0),  # Red
    alpha: float = 0.3
) -> Image.Image:
    """
    Highlight top-k% most important regions with colored overlay.
    
    Args:
        image: Original image
        heatmap: Attribution heatmap [H, W]
        top_k_percent: Percentage of pixels to highlight
        highlight_color: RGB color for highlighting (0-255)
        alpha: Transparency of highlight
        
    Returns:
        Image with highlighted regions as PIL Image
    """
    # Convert to numpy
    if isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()
    elif isinstance(image, Image.Image):
        image = np.array(image)
    
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Get top regions mask
    mask = get_top_regions_mask(heatmap, top_k_percent)
    
    # Create colored overlay
    overlay = image.copy()
    overlay[mask == 1] = (1 - alpha) * image[mask == 1] + alpha * np.array(highlight_color)
    overlay = overlay.astype(np.uint8)
    
    return Image.fromarray(overlay)
