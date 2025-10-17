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


class YFCC100MDataset(Dataset):
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
            'image': image,
            'lat': torch.tensor(float(record['latitude'])),
            'lon': torch.tensor(float(record['longitude'])),
            'idx': idx
        }





# Example usage
if __name__ == "__main__":
    # Example: Load dataset with custom transform
    from torch.utils.data import DataLoader
    
    
    # Create dataset with training transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = YFCC100MDataset(
        transform=train_transform,
        max_shards=1  # Limit to 1 shard for testing
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    # Test loading a batch
    for batch in dataloader:
        images = batch['image']
        latitudes = batch['latitude']
        longitudes = batch['longitude']
        ids = batch['id']
        
        print(f"Batch shape: {images.shape}")
        print(f"Latitudes: {latitudes[:5]}")
        print(f"Longitudes: {longitudes[:5]}")
        print(f"IDs: {ids[:5]}")
        break

