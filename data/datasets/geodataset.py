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


class MP16Dataset(Dataset):
    """
    PyTorch Dataset for loading geotagged images from MessagePack files.
    
    Each record contains:
        - 'image': binary image data (JPEG format)
        - 'id': image identifier
        - 'latitude': latitude coordinate
        - 'longitude': longitude coordinate
    """
    
    def __init__(
        self,
        transform: Optional[Callable],
        max_shards: Optional[int] = None
    ):
        """
        Args:
            data_dir: Directory containing the .msg shard files
            transform: Optional torchvision transform to apply to images
            max_shards: Optional limit on number of shards to load (useful for debugging)
        """
        self.data_dir = os.path.join(os.getenv("KAGGLEHUB_CACHE"), "datasets/habedi/large-dataset-of-geotagged-images")
        self.transform = transform
        
        # Find all shard files
        shard_path_pattern = os.path.join(self.data_dir, "**", "*.msg")
        self.shard_files = sorted(glob.glob(shard_path_pattern, recursive=True))
        
        if not self.shard_files:
            raise ValueError(f"No shard files found in {self.data_dir} matching pattern {"*.msg"}")
        
        if max_shards:
            self.shard_files = self.shard_files[:max_shards]
        
        print(f"Found {len(self.shard_files)} shard files")
        
        # Build index: list of (shard_idx, record_idx) tuples
        self.index = []
        self._build_index()
    
    def _build_index(self):
        """Build an index mapping dataset indices to (shard_file, record_position)."""
        print("Building dataset index...")
        for shard_idx, shard_file in tqdm(enumerate(self.shard_files), total=len(self.shard_files), desc="Processing shards"):
            with open(shard_file, 'rb') as f:
                unpacker = msgpack.Unpacker(f, raw=False)
                record_count = sum(1 for _ in unpacker)
            
            for record_idx in range(record_count):
                self.index.append((shard_idx, record_idx))
        
        print(f"Total records: {len(self.index)}")
    
    def __len__(self) -> int:
        return len(self.index)
    
    def _load_record(self, shard_idx: int, record_idx: int) -> Dict[str, Any]:
        """Load a specific record from a shard file."""
        shard_file = self.shard_files[shard_idx]
        
        with open(shard_file, 'rb') as f:
            unpacker = msgpack.Unpacker(f, raw=False)
            for i, record in enumerate(unpacker):
                if i == record_idx:
                    return record
        
        raise IndexError(f"Record {record_idx} not found in shard {shard_idx}")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset.

        Returns:
            Dictionary containing:
                - 'image': transformed image tensor
                - 'latitude': latitude value (float)
                - 'longitude': longitude value (float)
                - 'id': image identifier (string)
                - 'idx': dataset index (int) - can be used to retrieve this same item again
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        shard_idx, record_idx = self.index[idx]
        record = self._load_record(shard_idx, record_idx)

        # Decode image from bytes
        image_bytes = record['image']
        if isinstance(image_bytes, str):
            image_bytes = image_bytes.encode('latin1')

        image = Image.open(BytesIO(image_bytes)).convert('RGB')

        # Apply transform
        if self.transform:
            image = self.transform(image)

        return {
            'image_bytes': image_bytes,
            'image': image,
            'lat': torch.tensor(float(record['latitude'])),
            'lon': torch.tensor(float(record['longitude'])),
        }


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


