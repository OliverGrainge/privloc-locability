"""
Gradient-based attribution for geoprivacy risk prediction.

Implements gradient × input attribution to identify which image regions
most affect the model's geolocation risk score.
"""

from typing import Optional, Union, List, Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import gaussian_blur
import PIL


class GeoAttributionHeatmap: 
    """
    Generate gradient-based attribution heatmaps for geoprivacy risk.
    
    This class computes ∂(risk_score)/∂(image) to identify which spatial
    regions most affect whether an image is localizable.
    
    Args:
        model: Trained BinaryClassifier model (PyTorch Lightning module)
        transform: Image preprocessing transform (should match model's transform)
        device: Device to run computation on ('cpu', 'cuda', etc.)
        smooth_heatmap: Whether to apply Gaussian smoothing to reduce noise
        smooth_kernel_size: Kernel size for Gaussian smoothing (must be odd)
    """
    
    def __init__(
        self, 
        model: pl.LightningModule, 
        transform: transforms.Compose, 
        device: Optional[str] = "cpu",
        smooth_heatmap: bool = True,
        smooth_kernel_size: int = 15
    ):
        self.model = model.to(device)
        self.model.eval()  # Set to evaluation mode
        self.transform = transform
        self.device = device
        self.smooth_heatmap = smooth_heatmap
        self.smooth_kernel_size = smooth_kernel_size
        
        # Ensure kernel size is odd
        if self.smooth_kernel_size % 2 == 0:
            self.smooth_kernel_size += 1

    def _preprocess_image(self, image: Union[torch.Tensor, PIL.Image.Image, List[PIL.Image.Image]]) -> torch.Tensor:
        """
        Preprocess image(s) for model input.
        
        Args:
            image: Input image(s) - PIL Image, tensor, or list of PIL Images
            
        Returns:
            Preprocessed tensor [B, C, H, W]
        """
        if isinstance(image, PIL.Image.Image):
            image = self.transform(image).unsqueeze(0)  # [1, C, H, W]
        elif isinstance(image, list):
            image = torch.stack([self.transform(img) for img in image])
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)  # Add batch dimension
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        return image.to(self.device)

    def __call__(
        self, 
        image: Union[torch.Tensor, PIL.Image.Image, List[PIL.Image.Image]],
        return_risk_score: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate attribution heatmap showing which regions affect geolocation risk.
        
        Uses gradient × input attribution: ∂(risk_score)/∂(pixel) × pixel_value
        Higher values indicate regions that strongly affect localizability.
        
        Args:
            image: Input image(s) - PIL Image, tensor, or list of PIL Images
            return_risk_score: If True, also return the predicted risk scores
            
        Returns:
            If return_risk_score=False:
                Attribution heatmap [B, H, W] normalized to [0, 1]
            If return_risk_score=True:
                Tuple of (heatmap [B, H, W], risk_scores [B])
        """
        # 1. Preprocess image
        image_tensor = self._preprocess_image(image)
        batch_size = image_tensor.shape[0]
        
        # 2. Enable gradient computation
        image_tensor.requires_grad = True
        
        # 3. Forward pass through model
        with torch.set_grad_enabled(True):
            logits = self.model(image_tensor)  # [B]
            risk_scores = torch.sigmoid(logits)  # Risk probability in [0, 1]
        
        # 4. Compute gradients: ∂(risk_score)/∂(image)
        # We need gradients for each image in the batch
        self.model.zero_grad()
        if image_tensor.grad is not None:
            image_tensor.grad.zero_()
        
        # Compute gradients for the entire batch at once
        risk_scores.sum().backward()
        
        # 5. Compute attribution map using Gradient × Input
        # This gives "effective sensitivity" - which pixels have high gradient AND high values
        attribution = (image_tensor.grad * image_tensor).abs()  # [B, C, H, W]
        
        # Sum over color channels to get spatial attribution
        attribution = attribution.sum(dim=1)  # [B, H, W]
        
        # Detach from computation graph
        attribution = attribution.detach()
        
        # 6. Normalize to [0, 1] per image
        for i in range(batch_size):
            attr_min = attribution[i].min()
            attr_max = attribution[i].max()
            if attr_max > attr_min:
                attribution[i] = (attribution[i] - attr_min) / (attr_max - attr_min)
            else:
                # If attribution is constant (e.g., all zeros), keep as is
                attribution[i] = torch.zeros_like(attribution[i])
        
        # 7. Optional: Apply Gaussian smoothing to reduce noise
        if self.smooth_heatmap:
            attribution = attribution.unsqueeze(1)  # [B, 1, H, W] for gaussian_blur
            attribution = gaussian_blur(
                attribution, 
                kernel_size=self.smooth_kernel_size
            )
            attribution = attribution.squeeze(1)  # [B, H, W]
        
        # 8. Return heatmap (and optionally risk scores)
        if return_risk_score:
            return attribution, risk_scores.detach()
        else:
            return attribution
    
    def integrated_gradients(
        self,
        image: Union[torch.Tensor, PIL.Image.Image, List[PIL.Image.Image]],
        n_steps: int = 50,
        baseline: Optional[torch.Tensor] = None,
        return_risk_score: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate attribution using Integrated Gradients (more stable than plain gradients).
        
        Integrated Gradients averages gradients along a path from a baseline image
        (default: black image) to the actual image. This tends to produce more
        stable and theoretically grounded attributions.
        
        Args:
            image: Input image(s) - PIL Image, tensor, or list of PIL Images
            n_steps: Number of interpolation steps (more = more accurate but slower)
            baseline: Baseline image to integrate from (default: zeros/black image)
            return_risk_score: If True, also return the predicted risk scores
            
        Returns:
            If return_risk_score=False:
                Attribution heatmap [B, H, W] normalized to [0, 1]
            If return_risk_score=True:
                Tuple of (heatmap [B, H, W], risk_scores [B])
        """
        # 1. Preprocess image
        image_tensor = self._preprocess_image(image)
        batch_size = image_tensor.shape[0]
        
        # 2. Create baseline (default: black image)
        if baseline is None:
            baseline = torch.zeros_like(image_tensor)
        
        # 3. Generate interpolated images
        alphas = torch.linspace(0, 1, n_steps, device=self.device)
        
        # Collect gradients at each step
        gradients = []
        
        for alpha in alphas:
            # Interpolate between baseline and image
            interpolated = baseline + alpha * (image_tensor - baseline)
            interpolated.requires_grad = True
            
            # Forward pass
            logits = self.model(interpolated)
            risk_scores = torch.sigmoid(logits)
            
            # Backward pass
            self.model.zero_grad()
            if interpolated.grad is not None:
                interpolated.grad.zero_()
            
            risk_scores.sum().backward()
            
            # Store gradients
            gradients.append(interpolated.grad.detach())
        
        # 4. Average gradients across all interpolation steps
        avg_gradients = torch.stack(gradients).mean(dim=0)  # [B, C, H, W]
        
        # 5. Compute integrated gradients: avg_grad × (image - baseline)
        attribution = (avg_gradients * (image_tensor - baseline)).abs()  # [B, C, H, W]
        
        # Sum over color channels
        attribution = attribution.sum(dim=1)  # [B, H, W]
        
        # 6. Normalize to [0, 1] per image
        for i in range(batch_size):
            attr_min = attribution[i].min()
            attr_max = attribution[i].max()
            if attr_max > attr_min:
                attribution[i] = (attribution[i] - attr_min) / (attr_max - attr_min)
            else:
                attribution[i] = torch.zeros_like(attribution[i])
        
        # 7. Optional: Apply smoothing
        if self.smooth_heatmap:
            attribution = attribution.unsqueeze(1)
            attribution = gaussian_blur(attribution, kernel_size=self.smooth_kernel_size)
            attribution = attribution.squeeze(1)
        
        # 8. Get final risk scores
        with torch.no_grad():
            final_logits = self.model(image_tensor)
            final_risk_scores = torch.sigmoid(final_logits)
        
        if return_risk_score:
            return attribution, final_risk_scores
        else:
            return attribution
