from dotenv import load_dotenv
load_dotenv()
import os
import glob
from io import BytesIO
from typing import Optional, Callable, List, Dict, Any

import msgpack
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm



class OSV5MDataset(Dataset):
    """
    PyTorch Dataset for loading geotagged images from OSV5M using Hugging Face datasets.
    
    Each record contains:
        - 'image': PIL Image
        - 'latitude': latitude coordinate
        - 'longitude': longitude coordinate
        - Additional metadata fields
    """
    
    def __init__(
        self,
        transform: Optional[Callable],
        split: str = "train",
        streaming: bool = False,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            transform: Optional torchvision transform to apply to images
            split: Dataset split to load (e.g., 'train', 'test', 'validation')
            streaming: Whether to stream the dataset (useful for very large datasets)
            max_samples: Optional limit on number of samples to load (useful for debugging)
        """
        self.transform = transform
        self.streaming = streaming
        
        print(f"Loading OSV5M dataset (split: {split})...")
        
        # Load dataset from Hugging Face
        self.dataset = load_dataset(
            "osv5m/osv5m",
            split=split,
            streaming=streaming
        )
        
        # If not streaming and max_samples is specified, truncate
        if not streaming and max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
            print(f"Limited to {len(self.dataset)} samples")
        elif not streaming:
            print(f"Total records: {len(self.dataset)}")
        else:
            print("Streaming mode enabled")
    
    def __len__(self) -> int:
        if self.streaming:
            raise NotImplementedError("Length is not available in streaming mode")
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset.

        Returns:
            Dictionary containing:
                - 'image': transformed image tensor
                - 'latitude': latitude value (float)
                - 'longitude': longitude value (float)
                - 'image_bytes': raw image bytes
                - 'idx': dataset index (int) - can be used to retrieve this same item again
        """
        if self.streaming:
            raise NotImplementedError("Indexing is not available in streaming mode. Iterate over the dataset instead.")
        
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        record = self.dataset[idx]
        
        # Get the PIL Image (already decoded by HF datasets)
        image = record['image']
        
        # Convert to bytes if needed
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        # Apply transform
        if self.transform:
            image = self.transform(image)

        return {
            'image_bytes': image_bytes,
            'image': image,
            'lat': torch.tensor(float(record['latitude'])),
            'lon': torch.tensor(float(record['longitude'])),
        }




if __name__ == "__main__": 
    ds = OSV5MDataset(transform=transforms.ToTensor())
    batch = next(iter(ds))
    print(batch.keys())
    print(batch['image'].shape)
    print(batch['lat'])
    print(batch['lon'])