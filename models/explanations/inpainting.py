from typing import Optional, List, Union
import torch
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForInpainting
from accelerate import Accelerator


class Inpainting: 
    def __init__(
        self,
        default_prompt: str = "a photorealistic plausible replacement",
        model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        guidance_scale: float = 8.0,
        num_inference_steps: int = 20,
        strength: float = 0.99,
        target_size: int = 1024
    ):
        """
        Initialize the Inpainting model using Stable Diffusion XL.
        
        Args:
            default_prompt: Default prompt for inpainting if none provided.
            model_id: HuggingFace model ID for SDXL inpainting.
            guidance_scale: Guidance scale for diffusion (higher = more adherence to prompt).
            num_inference_steps: Number of denoising steps (15-30 works well).
            strength: Strength of inpainting (< 1.0 recommended).
            target_size: Target size for SDXL processing (1024 works best).
        """
        self.default_prompt = default_prompt
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.strength = strength
        self.target_size = target_size
        
        # Initialize device
        self.device = Accelerator().device
        
        # Load inpainting pipeline
        print(f"Loading SDXL inpainting model: {model_id}")
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(self.device)
        print("✓ SDXL inpainting model loaded successfully")
    
    def __call__(
        self, 
        image: Union[torch.Tensor, Image.Image],
        mask: Union[torch.Tensor, Image.Image],
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Inpaint the masked region of an image.
        
        Args:
            image: Input image as PIL Image or torch Tensor.
            mask: Binary mask as torch Tensor [H, W] or PIL Image.
                  1/255 = region to inpaint, 0 = keep original.
            prompt: Text prompt describing what to generate in masked region.
            negative_prompt: What to avoid in the generation.
            seed: Random seed for reproducibility.
            
        Returns:
            PIL Image with inpainted region.
        """
        # Use default prompt if none provided
        if prompt is None:
            prompt = self.default_prompt
        
        # Convert inputs to PIL Images
        pil_image = self._to_pil_image(image)
        pil_mask = self._mask_to_pil_image(mask, pil_image.size)
        
        # Store original size for later
        original_size = pil_image.size
        
        # Resize to target size for SDXL (works best at 1024x1024)
        pil_image_resized = pil_image.resize((self.target_size, self.target_size), Image.LANCZOS)
        pil_mask_resized = pil_mask.resize((self.target_size, self.target_size), Image.LANCZOS)
        
        # Setup generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Run inpainting
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pil_image_resized,
            mask_image=pil_mask_resized,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            strength=self.strength,
            generator=generator,
        ).images[0]
        
        # Resize back to original size if needed
        if original_size != (self.target_size, self.target_size):
            result = result.resize(original_size, Image.LANCZOS)
        
        return result
    
    def inpaint_multiple_regions(
        self,
        image: Union[torch.Tensor, Image.Image],
        masks: torch.Tensor,
        labels: List[str],
        prompt_template: str = "replace with {label}",
        combine_masks: bool = False
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Inpaint multiple regions, either separately or combined.
        
        Args:
            image: Input image.
            masks: Tensor of masks [N, H, W].
            labels: List of labels corresponding to each mask.
            prompt_template: Template for generating prompts. Use {label} placeholder.
            combine_masks: If True, combines all masks and inpaints once.
                          If False, inpaints each region separately.
        
        Returns:
            Single inpainted image if combine_masks=True, otherwise list of images.
        """
        if combine_masks:
            # Combine all masks into one
            combined_mask = self._combine_masks(masks)
            
            # Create a prompt that mentions all labels
            labels_str = ", ".join(set(labels))
            prompt = prompt_template.format(label=labels_str)
            
            return self(image, combined_mask, prompt=prompt)
        else:
            # Inpaint each region separately
            results = []
            current_image = self._to_pil_image(image)
            
            for mask, label in zip(masks, labels):
                prompt = prompt_template.format(label=label)
                current_image = self(current_image, mask, prompt=prompt)
                results.append(current_image.copy())
            
            return results
    
    def _to_pil_image(self, image: Union[torch.Tensor, Image.Image]) -> Image.Image:
        """Convert various image formats to PIL Image."""
        if isinstance(image, Image.Image):
            return image.convert('RGB')
        elif isinstance(image, torch.Tensor):
            # Assume tensor is in [C, H, W] format
            if image.ndim == 4:
                image = image[0]  # Take first image if batch
            
            # Convert to [H, W, C]
            if image.shape[0] == 3:  # [C, H, W]
                image = image.permute(1, 2, 0)
            
            # Normalize to [0, 255] if needed
            if image.max() <= 1.0:
                image = image * 255
            
            # Convert to numpy and PIL
            image_np = image.cpu().byte().numpy()
            return Image.fromarray(image_np, mode='RGB')
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
    
    def _mask_to_pil_image(
        self, 
        mask: Union[torch.Tensor, Image.Image],
        target_size: tuple
    ) -> Image.Image:
        """
        Convert mask to PIL Image format expected by inpainting pipeline.
        
        Args:
            mask: Binary mask as tensor [H, W] or PIL Image.
            target_size: (width, height) to resize mask to.
            
        Returns:
            PIL Image in 'L' (grayscale) mode with values 0-255.
        """
        if isinstance(mask, Image.Image):
            # Already PIL Image, just ensure correct mode and size
            mask_pil = mask.convert('L')
        elif isinstance(mask, torch.Tensor):
            # Convert tensor to PIL Image
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]  # Remove channel dimension if present
            
            # Ensure binary values (0 or 1)
            mask = torch.clamp(mask, 0, 1)
            
            # Convert to 0-255 range
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_np, mode='L')
        else:
            raise TypeError(f"Unsupported mask type: {type(mask)}")
        
        # Resize to match image size
        if mask_pil.size != target_size:
            mask_pil = mask_pil.resize(target_size, Image.BILINEAR)
        
        return mask_pil
    
    def _combine_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Combine multiple binary masks into one.
        
        Args:
            masks: Tensor of shape [N, H, W]
            
        Returns:
            Combined mask of shape [H, W]
        """
        combined = torch.sum(masks, dim=0)
        combined = torch.clamp(combined, 0, 1)  # Ensure binary
        return combined