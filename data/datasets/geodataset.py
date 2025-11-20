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
        max_shards: Optional[int] = 5
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





import os
import pandas as pd
from PIL import Image
from typing import Optional, Callable, Dict, Any
import torch
from torch.utils.data import Dataset

class YFCC4kDataset(Dataset):
    """
    PyTorch Dataset for loading YFCC4K geotagged images.
    
    Dataset structure:
        datafolder/
            yfcc4k/
                yfcc4k/
                    *.jpg
                yfcc4k.txt
    """
    
    def __init__(
        self,
        transform: Optional[Callable] = None,
        **kwargs
    ):
        """
        Args:
            transform: Optional torchvision transform to apply to images
        """
        # Construct base path
        self.data_dir = os.path.join(
            os.getenv("KAGGLEHUB_CACHE"), 
            "datasets/lbgan2000/img2gps-yfcc4k/versions/7"
        )
        
        # Path to images and metadata
        self.image_dir = os.path.join(self.data_dir, "yfcc4k", "yfcc4k")
        self.metadata_file = os.path.join(self.data_dir, "yfcc4k/yfcc4k.txt")
        
        self.transform = transform
        
        # Verify paths exist
        if not os.path.exists(self.metadata_file):
            raise ValueError(f"Metadata file not found: {self.metadata_file}")
        if not os.path.exists(self.image_dir):
            raise ValueError(f"Image directory not found: {self.image_dir}")
        
        # Load metadata
        self._load_metadata()
        
        print(f"Loaded {len(self.metadata)} images from YFCC4K dataset")
    
    def _load_metadata(self):
        """Load and parse the metadata file."""
        # Column names based on your sample data
        columns = [
            'line_number', 'photo_id', 'hash', 'user_nsid', 'user_nickname',
            'date_taken', 'date_uploaded', 'capture_device', 'title',
            'description', 'user_tags', 'machine_tags', 'longitude', 'latitude',
            'accuracy', 'photo_url', 'download_url', 'license', 'license_url',
            'server_id', 'farm_id', 'secret', 'secret_original', 'extension',
            'video'
        ]
        
        # Read the tab-separated file
        self.metadata = pd.read_csv(
            self.metadata_file,
            sep='\t',
            names=columns,
            header=None
        )
        
        # Create image filename from photo_id
        self.metadata['image_filename'] = self.metadata['photo_id'].astype(str) + '.jpg'
        
        # Filter out any rows where coordinates are missing
        self.metadata = self.metadata.dropna(subset=['latitude', 'longitude'])
        
        # Reset index after filtering
        self.metadata = self.metadata.reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset.
        
        Returns:
            Dictionary containing:
                - 'image_bytes': raw image bytes
                - 'image': transformed image tensor
                - 'lat': latitude value (tensor)
                - 'lon': longitude value (tensor)
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        # Get metadata for this image
        row = self.metadata.iloc[idx]
        
        # Construct full image path
        image_path = os.path.join(self.image_dir, row['image_filename'])
        
        # Load image bytes
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {str(e)}")
        
        # Decode image from bytes
        from io import BytesIO
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return {
            'image_bytes': image_bytes,
            'image': image,
            'lat': torch.tensor(float(row['latitude'])),
            'lon': torch.tensor(float(row['longitude'])),
        }



class Im2GPS3kDataset(Dataset):
    """
    PyTorch Dataset for loading im2gps3k geotagged images.
    
    Dataset structure:
        datafolder/
            im2gps3ktest/
                im2gps3ktest/
                    *.jpg
            im2gps3k_places365.csv
    """
    
    def __init__(
        self,
        transform: Optional[Callable] = None,
        **kwargs
    ):
        """
        Args:
            transform: Optional torchvision transform to apply to images
        """
        # Construct base path
        self.data_dir = os.path.join(
            os.getenv("KAGGLEHUB_CACHE"), 
            "datasets/lbgan2000/img2gps-yfcc4k/versions/7/"
        )
        
        # Path to images and metadata
        self.image_dir = os.path.join(self.data_dir, "im2gps3ktest", "im2gps3ktest")
        self.metadata_file = os.path.join(self.data_dir, "im2gps3k_places365.csv")
        
        self.transform = transform
        
        # Verify paths exist
        if not os.path.exists(self.metadata_file):
            raise ValueError(f"Metadata file not found: {self.metadata_file}")
        if not os.path.exists(self.image_dir):
            raise ValueError(f"Image directory not found: {self.image_dir}")
        
        # Load metadata
        self._load_metadata()
        
        print(f"Loaded {len(self.metadata)} images from im2gps3k dataset")
    
    def _load_metadata(self):
        """Load and parse the metadata CSV file."""
        # Read the CSV file with proper header handling
        # The CSV has columns: name,AUTHOR,LAT,LON,S3_Label,S16_Label,S365_Label,Prob_indoor,Prob_natural,Prob_urban
        self.metadata = pd.read_csv(
            self.metadata_file,
            header=0  # Use the first row as header
        )
        
        # Rename columns to match our expected format
        self.metadata = self.metadata.rename(columns={
            'name': 'image_filename',
            'AUTHOR': 'user_id', 
            'LAT': 'latitude',
            'LON': 'longitude'
        })
        
        # Filter out any rows where coordinates are missing
        self.metadata = self.metadata.dropna(subset=['latitude', 'longitude'])
        
        # Reset index after filtering
        self.metadata = self.metadata.reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset.
        
        Returns:
            Dictionary containing:
                - 'image_bytes': raw image bytes
                - 'image': transformed image tensor
                - 'lat': latitude value (tensor)
                - 'lon': longitude value (tensor)
                - 'user_id': user ID string
                - 'filename': image filename
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        # Get metadata for this image
        row = self.metadata.iloc[idx]
        
        # Construct full image path
        image_path = os.path.join(self.image_dir, row['image_filename'])
        
        # Load image bytes
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {str(e)}")
        
        # Decode image from bytes
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return {
            'image_bytes': image_bytes,
            'image': image,
            'lat': torch.tensor(float(row['latitude'])),
            'lon': torch.tensor(float(row['longitude'])),
        }