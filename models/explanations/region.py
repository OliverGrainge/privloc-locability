from typing import Optional, List, Dict, Any, Union
import torch
from PIL import Image
from transformers import (
    AutoProcessor, 
    AutoModelForZeroShotObjectDetection,
    SamModel,
    SamProcessor
)
from accelerate import Accelerator


class EditableRegionExtraction: 
    def __init__(
        self, 
        text_labels: Optional[List[str]] = None,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        sam_model_id: str = "facebook/sam-vit-base",
        threshold: float = 0.4,
        text_threshold: float = 0.3,
        use_sam: bool = True
    ):
        """
        Initialize the Editable Region Extraction model using Grounding DINO + SAM.
        
        Args:
            text_labels: List of text labels to detect. If None, uses default labels.
            model_id: HuggingFace model ID for Grounding DINO.
            sam_model_id: HuggingFace model ID for SAM (Segment Anything Model).
            threshold: Detection confidence threshold.
            text_threshold: Text matching threshold.
            use_sam: If True, uses SAM for precise segmentation. If False, uses bounding boxes.
        """
        self.text_labels = text_labels if text_labels is not None else self._default_text_labels()
        self.threshold = threshold
        self.text_threshold = text_threshold
        self.use_sam = use_sam
        
        # Initialize device
        self.device = Accelerator().device
        
        # Initialize Grounding DINO
        print(f"Loading Grounding DINO model: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        
        # Initialize SAM if requested
        if self.use_sam:
            print(f"Loading SAM model: {sam_model_id}")
            self.sam_processor = SamProcessor.from_pretrained(sam_model_id)
            self.sam_model = SamModel.from_pretrained(sam_model_id).to(self.device)
            print("✓ SAM model loaded successfully")
        else:
            self.sam_processor = None
            self.sam_model = None
        
    def _default_text_labels(self) -> List[str]:
        """Default text labels for region detection."""
        return [
            "building",
            "tree",
            "car",
            "sign",
            "animal",
            "plant",
            "object",
            "sky",
            "water",
        ]

    def __call__(
        self, 
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        return_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Extract regions from the image based on text labels and return binary masks.
        
        Args:
            image: Input image(s) as PIL Image, torch Tensor, or list of PIL Images.
            return_dict: If True, returns dict with masks and metadata. If False, returns only masks tensor.
            
        Returns:
            If return_dict=False (default):
                Tensor of shape [N, H, W] where N is number of detected regions,
                H and W are image height and width. Each mask is binary (0 or 1).
            If return_dict=True:
                Dictionary containing:
                    - 'masks': Tensor of shape [N, H, W]
                    - 'labels': List of label strings
                    - 'scores': Tensor of confidence scores [N]
                    - 'image_size': Tuple of (width, height)
        """
        # Convert torch tensor to PIL Image if needed
        if isinstance(image, torch.Tensor):
            if image.ndim == 4:
                # Batch of images - only process first one for now
                img = self._tensor_to_pil(image[0])
            else:
                # Single image
                img = self._tensor_to_pil(image)
        elif isinstance(image, Image.Image):
            img = image
        elif isinstance(image, list):
            # Process first image if list
            img = image[0]
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Prepare text labels in the format expected by the processor
        text_labels_formatted = [[label for label in self.text_labels]]
        
        # Process inputs for Grounding DINO
        inputs = self.processor(
            images=img, 
            text=text_labels_formatted, 
            return_tensors="pt"
        ).to(self.model.device)
        
        # Run Grounding DINO inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results to get bounding boxes
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.threshold,
            text_threshold=self.text_threshold,
            target_sizes=[img.size[::-1]]
        )
        
        # Extract bounding boxes
        result = results[0]
        height, width = img.size[1], img.size[0]  # PIL size is (width, height)
        
        if len(result["boxes"]) == 0:
            # No detections - return empty tensors
            masks = torch.zeros((0, height, width), dtype=torch.float32, device=self.device)
            scores = torch.zeros((0,), dtype=torch.float32, device=self.device)
            labels_list = []
        elif self.use_sam and self.sam_model is not None:
            # Use SAM to generate precise masks from bounding boxes
            masks, labels_list, scores = self._generate_sam_masks(
                img, 
                result["boxes"], 
                result["scores"], 
                result["labels"]
            )
        else:
            # Fall back to rectangular masks from bounding boxes
            masks_list = []
            labels_list = []
            scores_list = []
            
            for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
                mask = self._box_to_mask(box, height, width)
                masks_list.append(mask)
                labels_list.append(label)
                scores_list.append(score)
            
            masks = torch.stack(masks_list, dim=0)
            scores = torch.stack(scores_list, dim=0)
        
        if return_dict:
            return {
                'masks': masks,
                'labels': labels_list,
                'scores': scores,
                'image_size': img.size  # (width, height)
            }
        else:
            return masks
    
    def _generate_sam_masks(
        self, 
        image: Image.Image, 
        boxes: torch.Tensor, 
        scores: torch.Tensor, 
        labels: List[str]
    ) -> tuple[torch.Tensor, List[str], torch.Tensor]:
        """
        Generate precise segmentation masks using SAM from bounding boxes.
        
        Args:
            image: PIL Image
            boxes: Bounding boxes tensor [N, 4] in format [x_min, y_min, x_max, y_max]
            scores: Confidence scores tensor [N]
            labels: List of label strings
            
        Returns:
            Tuple of (masks, labels, scores) where masks is [N, H, W]
        """
        # Prepare bounding boxes for SAM (needs to be in list format)
        # SAM expects boxes in [[x_min, y_min, x_max, y_max], ...]
        input_boxes = [[box.tolist()] for box in boxes]
        
        masks_list = []
        valid_labels = []
        valid_scores = []
        
        # Process each box individually with SAM
        for idx, (box_list, score, label) in enumerate(zip(input_boxes, scores, labels)):
            # Prepare inputs for SAM
            sam_inputs = self.sam_processor(
                image, 
                input_boxes=[box_list],  # SAM expects nested list
                return_tensors="pt"
            ).to(self.device)
            
            # Run SAM inference
            with torch.no_grad():
                sam_outputs = self.sam_model(**sam_inputs)
            
            # Post-process SAM outputs to get masks
            sam_masks = self.sam_processor.image_processor.post_process_masks(
                sam_outputs.pred_masks.cpu(),
                sam_inputs["original_sizes"].cpu(),
                sam_inputs["reshaped_input_sizes"].cpu()
            )[0]  # Get first (and only) image's masks
            
            # SAM returns masks of shape [1, num_masks_per_box, H, W]
            # We take the first mask (usually the best one)
            mask = sam_masks[0, 0]  # [H, W]
            
            # Convert to binary mask (threshold at 0.5)
            mask = (mask > 0.5).float().to(self.device)
            
            masks_list.append(mask)
            valid_labels.append(label)
            valid_scores.append(score)
        
        # Stack all masks
        if len(masks_list) > 0:
            masks = torch.stack(masks_list, dim=0)  # [N, H, W]
            scores_tensor = torch.stack(valid_scores, dim=0)
        else:
            height, width = image.size[1], image.size[0]
            masks = torch.zeros((0, height, width), dtype=torch.float32, device=self.device)
            scores_tensor = torch.zeros((0,), dtype=torch.float32, device=self.device)
        
        return masks, valid_labels, scores_tensor
    
    def _box_to_mask(self, box: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Convert bounding box to binary mask.
        
        Args:
            box: Bounding box tensor [x_min, y_min, x_max, y_max]
            height: Image height
            width: Image width
            
        Returns:
            Binary mask tensor [H, W]
        """
        mask = torch.zeros((height, width), dtype=torch.float32, device=self.device)
        
        # Extract box coordinates and clamp to image bounds
        x_min = int(torch.clamp(box[0], 0, width - 1))
        y_min = int(torch.clamp(box[1], 0, height - 1))
        x_max = int(torch.clamp(box[2], 0, width))
        y_max = int(torch.clamp(box[3], 0, height))
        
        # Fill the bounding box region with 1s
        mask[y_min:y_max, x_min:x_max] = 1.0
        
        return mask
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert a torch tensor to PIL Image."""
        # Assume tensor is in [C, H, W] format with values in [0, 1] or [0, 255]
        if tensor.max() <= 1.0:
            tensor = tensor * 255
        tensor = tensor.byte()
        
        # Convert to [H, W, C] for PIL
        if tensor.ndim == 3:
            tensor = tensor.permute(1, 2, 0)
        
        # Convert to numpy and then to PIL
        import numpy as np
        array = tensor.cpu().numpy()
        return Image.fromarray(array)