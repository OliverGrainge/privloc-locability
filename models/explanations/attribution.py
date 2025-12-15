"""
Attribution methods for geoprivacy risk prediction.

Implements multiple attribution methods:
- Gradient × input attribution
- Integrated gradients
- Attention rollout (for Vision Transformers like DINOv2)
- Raw gradients
"""

from typing import Optional, Union, List, Tuple, Dict
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import gaussian_blur
import PIL
import numpy as np
from models.binaryclassifer import BinaryClassifier


class GeoAttributionHeatmap: 
    def __init__(self, model: BinaryClassifier, transform: transforms.Compose):
        self.model = model
        self.transform = transform
        self.device = self._get_device()
        self.model.to(self.device)
        self.model.eval()

    def _get_device(self) -> str:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, image: Union[torch.Tensor, PIL.Image.Image, List[PIL.Image.Image]], 
                 head_fusion: str = "mean", discard_ratio: float = 0.1) -> torch.Tensor:
        """
        Compute attention rollout saliency map.
        
        Args:
            image: Input image(s) to compute attribution for
            head_fusion: How to combine attention heads ('mean', 'max', or 'min')
            discard_ratio: Ratio of lowest attentions to discard for stability
            
        Returns:
            Saliency map of shape [batch_size, height, width] or [height, width] if single image
        """
        image = self._preprocess_image(image)
        image = image.to(self.device)
        
        # Get attention maps from all layers
        attention_maps = self._extract_attention_maps(image)
        
        # Compute attention rollout
        rollout = self._compute_attention_rollout(attention_maps, head_fusion, discard_ratio)
        
        # Convert to spatial saliency map
        saliency = self._rollout_to_saliency(rollout, image.shape[-2:])
        
        return saliency.squeeze() if saliency.shape[0] == 1 else saliency

    def _preprocess_image(self, image: Union[torch.Tensor, PIL.Image.Image, List[PIL.Image.Image]]) -> torch.Tensor:
        """Convert image to tensor and apply transforms."""
        if isinstance(image, torch.Tensor):
            return image
        
        if isinstance(image, PIL.Image.Image):
            return self.transform(image).unsqueeze(0)
        
        if isinstance(image, list):
            return torch.stack([self.transform(img) for img in image])
        
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _extract_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract attention maps from all transformer blocks.
        
        Returns:
            List of attention tensors, each of shape [batch, heads, tokens, tokens]
        """
        attention_maps = []
        
        # Hook to capture attention weights
        def hook_fn(module, input, output):
            # For DINOv2/ViT, attention weights are computed in the attention module
            # We need to compute them from the query, key matrices
            B, N, C = input[0].shape
            qkv = module.qkv(input[0]).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)
            
            attention_maps.append(attn.detach())
        
        # Register hooks on attention blocks starting from layer 3
        hooks = []
        backbone_model = self.model.model.backbone.model
        start_layer = 3  # Skip layers 0, 1, 2
        for i, block in enumerate(backbone_model.blocks):
            if i >= start_layer:
                hook = block.attn.register_forward_hook(hook_fn)
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_maps
    
    def _compute_attention_rollout(self, attention_maps: List[torch.Tensor], 
                                   head_fusion: str = "mean", 
                                   discard_ratio: float = 0.1) -> torch.Tensor:
        """
        Compute attention rollout from layer attention maps.
        
        Args:
            attention_maps: List of attention tensors [batch, heads, tokens, tokens]
            head_fusion: How to fuse attention heads
            discard_ratio: Ratio of lowest attentions to discard
            
        Returns:
            Rolled attention of shape [batch, tokens, tokens]
        """
        batch_size = attention_maps[0].shape[0]
        num_tokens = attention_maps[0].shape[-1]
        
        # Fuse attention heads for each layer
        fused_attentions = []
        for attn in attention_maps:
            if head_fusion == "mean":
                fused = attn.mean(dim=1)  # [batch, tokens, tokens]
            elif head_fusion == "max":
                fused = attn.max(dim=1)[0]
            elif head_fusion == "min":
                fused = attn.min(dim=1)[0]
            else:
                raise ValueError(f"Unknown head_fusion: {head_fusion}")
            
            # Discard lowest attentions for stability
            if discard_ratio > 0:
                flat = fused.view(batch_size, -1)
                threshold = flat.quantile(discard_ratio, dim=-1, keepdim=True)
                fused = torch.where(
                    fused < threshold.view(batch_size, 1, 1),
                    torch.zeros_like(fused),
                    fused
                )
            
            # Add identity matrix to account for residual connections
            I = torch.eye(num_tokens, device=fused.device).unsqueeze(0)
            fused = fused + I
            
            # Re-normalize
            fused = fused / fused.sum(dim=-1, keepdim=True)
            
            fused_attentions.append(fused)
        
        # Recursively multiply attention matrices (rollout)
        rollout = fused_attentions[0]
        for attn in fused_attentions[1:]:
            rollout = torch.matmul(attn, rollout)
        
        return rollout
    
    def _rollout_to_saliency(self, rollout: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Convert attention rollout to spatial saliency map.
        
        Args:
            rollout: Attention rollout [batch, tokens, tokens]
            image_size: (height, width) of input image
            
        Returns:
            Saliency map [batch, height, width]
        """
        # Take attention from CLS token to all patches
        # CLS token is at index 0
        cls_attention = rollout[:, 0, 1:]  # [batch, num_patches]
        
        # Reshape to spatial grid
        # For DINOv2/ViT with patch size 14, calculate grid size
        patch_size = 14
        h_patches = image_size[0] // patch_size
        w_patches = image_size[1] // patch_size
        
        saliency = cls_attention.view(-1, h_patches, w_patches)
        
        # Upsample to image resolution
        saliency = F.interpolate(
            saliency.unsqueeze(1),
            size=image_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        # Remove outliers using percentile-based thresholding
        # Clip extreme high values to reduce single hotspots
        batch_size = saliency.shape[0]
        for i in range(batch_size):
            sal = saliency[i]
            # Calculate 95th percentile
            percentile_98 = torch.quantile(sal.flatten(), 0.98)
            # Clip values above 95th percentile
            saliency[i] = torch.clamp(sal, max=percentile_98)
        
        # Apply Gaussian smoothing to create more coherent regions
        saliency = gaussian_blur(saliency.unsqueeze(1), kernel_size=21, sigma=3.0).squeeze(1)
        
        # Normalize to [0, 1]
        saliency = (saliency - saliency.amin(dim=(1, 2), keepdim=True)) / (
            saliency.amax(dim=(1, 2), keepdim=True) - saliency.amin(dim=(1, 2), keepdim=True) + 1e-8
        )
        
        return saliency
    