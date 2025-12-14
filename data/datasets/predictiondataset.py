import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
from typing import Optional, Callable, Union, List
from pathlib import Path
import glob
from tqdm import tqdm 
from torchvision import transforms
from torch.utils.data._utils.collate import default_collate


def prediction_collate_fn(batch):
    """
    Custom collate function for prediction datasets that handles PIL Images.
    
    PIL Images are kept as lists instead of being stacked, since they can't be batched.
    All other fields use the default collate function.
    """
    from PIL import Image as PILImage
    
    # Separate PIL Images from other data
    pil_image_key = 'pil_image'
    pil_images = []
    other_data = []
    
    for item in batch:
        if pil_image_key in item:
            pil_images.append(item.pop(pil_image_key))
        other_data.append(item)
    
    # Use default collate for the rest
    batch_dict = default_collate(other_data)
    
    # Add PIL images as a list
    if pil_images:
        batch_dict[pil_image_key] = pil_images
    
    return batch_dict


class PredictionBaseDataset(Dataset):
    """
    Base PyTorch Dataset for loading prediction results from parquet files.
    
    The parquet files contain:
        - 'image_bytes': binary image data (JPEG format)
        - 'true_lat': true latitude coordinate
        - 'true_lon': true longitude coordinate
        - 'pred_lat': predicted latitude coordinate
        - 'pred_lon': predicted longitude coordinate
        - 'embedding': image embedding (optional, if generated with embeddings=True)
    """
    
    def __init__(
        self,
        parquet_files: List[Union[str, Path]],
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            parquet_files: List of parquet file paths (shard file paths)
            transform: Optional torchvision transform to apply to images
        """
        self.transform = transform
        self.parquet_files = [Path(f) for f in parquet_files]

        # Initialize lazy loading state
        self.current_file_idx = None
        self.current_df = None
        self.file_start_indices = []
        self.total_samples = 0
        
        # Prefetching state
        self.prefetch_next = False  # Disable to save memory
        self.next_file_df = None
        self.next_file_idx = None
        
        # Embedding state
        self.has_embeddings = False
        
        # Calculate file start indices and total samples without loading data
        print(f"Scanning {len(self.parquet_files)} parquet file(s) for metadata")
        for i, file_path in enumerate(self.parquet_files):
            # Read just the metadata to get row count
            df_meta = pd.read_parquet(file_path, engine='pyarrow')
            file_rows = len(df_meta)
            print(f"File {i + 1}/{len(self.parquet_files)}: {file_path} ({file_rows} rows)")
            
            self.file_start_indices.append(self.total_samples)
            self.total_samples += file_rows
        
        print(f"Total samples across all files: {self.total_samples}")
        
        # Verify required columns exist by checking the first file
        if self.parquet_files:
            sample_df = pd.read_parquet(self.parquet_files[0])
            required_columns = ['image_bytes', 'true_lat', 'true_lon', 'pred_lat', 'pred_lon']
            missing_columns = [col for col in required_columns if col not in sample_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in parquet file: {missing_columns}")
            
            # Check if embeddings are present
            if 'embedding' in sample_df.columns:
                self.has_embeddings = True
                print("Embeddings detected in parquet files")
            else:
                print("No embeddings found in parquet files")

    def _load_file_if_needed(self, file_idx: int):
        """Load a parquet file if it's not currently loaded."""
        if self.current_file_idx != file_idx:
            # Check if we have prefetched data for this file
            if self.next_file_idx == file_idx and self.next_file_df is not None:
                print(f"Using prefetched Shard {file_idx + 1}/{len(self.parquet_files)}: {self.parquet_files[file_idx]}")
                self.current_df = self.next_file_df
                self.current_file_idx = file_idx
                self.next_file_df = None
                self.next_file_idx = None
            else:
                # Clear previous file from memory
                if self.current_file_idx is not None and self.current_df is not None:
                    del self.current_df
                    import gc
                    gc.collect()
                    # Clear CUDA cache if available
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
                
                print(f"Loading Shard {file_idx + 1}/{len(self.parquet_files)}: {self.parquet_files[file_idx]}")
                self.current_df = pd.read_parquet(self.parquet_files[file_idx])
                self.current_file_idx = file_idx
            
            # Prefetch next file if available
            if self.prefetch_next and file_idx + 1 < len(self.parquet_files):
                self._prefetch_next_file(file_idx)
    
    def _prefetch_next_file(self, current_file_idx: int):
        """Prefetch the next file in background."""
        next_file_idx = current_file_idx + 1
        if next_file_idx < len(self.parquet_files) and self.next_file_idx != next_file_idx:
            try:
                print(f"Prefetching Shard {next_file_idx + 1}/{len(self.parquet_files)}: {self.parquet_files[next_file_idx]}")
                self.next_file_df = pd.read_parquet(self.parquet_files[next_file_idx])
                self.next_file_idx = next_file_idx
            except Exception as e:
                print(f"Failed to prefetch file {next_file_idx}: {e}")
                self.next_file_df = None
                self.next_file_idx = None
    
    def get_file_samples(self, file_idx: int) -> List[dict]:
        """
        Get all samples from a specific parquet file.
        This is more efficient than random access when you need all samples from a file.
        
        Args:
            file_idx: Index of the parquet file to load
            
        Returns:
            List of sample dictionaries
        """
        if file_idx < 0 or file_idx >= len(self.parquet_files):
            raise IndexError(f"File index {file_idx} out of range for {len(self.parquet_files)} files")
        
        # Load the file
        self._load_file_if_needed(file_idx)
        
        samples = []
        for i in range(len(self.current_df)):
            global_idx = self.file_start_indices[file_idx] + i
                
            row = self.current_df.iloc[i]
            
            # Decode image from bytes
            image_bytes = row['image_bytes']
            if isinstance(image_bytes, str):
                image_bytes = image_bytes.encode('latin1')
            
            pil_image = Image.open(BytesIO(image_bytes)).convert('RGB')
            
            # Apply transform if provided, otherwise convert to tensor
            if self.transform:
                image = self.transform(pil_image)
            else:
                # Default: convert PIL image to tensor
                image = transforms.ToTensor()(pil_image)
            
            sample = {
                'image': image,
                'pil_image': pil_image,  # Always include PIL image for plonk
                'true_lat': torch.tensor(float(row['true_lat'])),
                'true_lon': torch.tensor(float(row['true_lon'])),
                'pred_lat': torch.tensor(float(row['pred_lat'])),
                'pred_lon': torch.tensor(float(row['pred_lon'])),
                'idx': torch.tensor(global_idx)
            }
            
            # Add embedding if available
            if self.has_embeddings and 'embedding' in row:
                # Convert bytes back to numpy array
                import numpy as np
                embedding_bytes = row['embedding']
                if isinstance(embedding_bytes, bytes):
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    sample['embedding'] = torch.tensor(embedding)
            
            samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.total_samples
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing:
                - 'image': transformed image tensor
                - 'true_lat': true latitude value (float)
                - 'true_lon': true longitude value (float)
                - 'pred_lat': predicted latitude value (float)
                - 'pred_lon': predicted longitude value (float)
                - 'idx': dataset index (int)
                - 'embedding': image embedding tensor (if available)
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        # Find which file contains this index
        file_idx = 0
        for i, start_idx in enumerate(self.file_start_indices):
            if i + 1 < len(self.file_start_indices):
                next_start = self.file_start_indices[i + 1]
            else:
                next_start = self.total_samples
            
            if start_idx <= idx < next_start:
                file_idx = i
                break
        
        # Load the file if needed
        self._load_file_if_needed(file_idx)
        
        # Calculate the local index within the current file
        local_idx = idx - self.file_start_indices[file_idx]
        
        # Get the row data
        row = self.current_df.iloc[local_idx]
        
        # Decode image from bytes
        image_bytes = row['image_bytes']
        if isinstance(image_bytes, str):
            image_bytes = image_bytes.encode('latin1')
        
        pil_image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Apply transform if provided, otherwise resize and convert to tensor
        if self.transform:
            image = self.transform(pil_image)
        else:
            # Default: resize to fixed size and convert to tensor for batching
            # Use 224x224 as a standard size for vision models
            default_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            image = default_transform(pil_image)
        
        result = {
            'image': image,
            'pil_image': pil_image, 
            'true_lat': torch.tensor(float(row['true_lat'])),
            'true_lon': torch.tensor(float(row['true_lon'])),
            'pred_lat': torch.tensor(float(row['pred_lat'])),
            'pred_lon': torch.tensor(float(row['pred_lon'])),
            'idx': torch.tensor(idx)
        }
        
        # Add embedding if available
        if self.has_embeddings and 'embedding' in row:
            # Convert bytes back to numpy array
            import numpy as np
            embedding_bytes = row['embedding']
            if isinstance(embedding_bytes, bytes):
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                result['embedding'] = torch.tensor(embedding)
        
        return result
    
    def get_stats(self) -> dict:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        # Get file information
        parquet_files = [str(f) for f in self.parquet_files]
        
        # For coordinate ranges, we need to load all files and compute statistics
        # This is expensive but necessary for accurate stats
        print("Computing coordinate statistics across all files...")
        all_lats = []
        all_lons = []
        all_pred_lats = []
        all_pred_lons = []
        
        for file_path in self.parquet_files:
            df = pd.read_parquet(file_path)
            all_lats.extend(df['true_lat'].tolist())
            all_lons.extend(df['true_lon'].tolist())
            all_pred_lats.extend(df['pred_lat'].tolist())
            all_pred_lons.extend(df['pred_lon'].tolist())
        
        columns = ['image_bytes', 'true_lat', 'true_lon', 'pred_lat', 'pred_lon']
        if self.has_embeddings:
            columns.append('embedding')
        
        return {
            'total_samples': self.total_samples,
            'parquet_files': parquet_files,
            'num_shards': len(parquet_files),
            'columns': columns,
            'has_embeddings': self.has_embeddings,
            'lat_range': (min(all_lats), max(all_lats)),
            'lon_range': (min(all_lons), max(all_lons)),
            'pred_lat_range': (min(all_pred_lats), max(all_pred_lats)),
            'pred_lon_range': (min(all_pred_lons), max(all_pred_lons))
        }


class Mp16PredictionDataset(PredictionBaseDataset):
    """
    Prediction dataset for MP16 with train/val split support.
    
    MP16 is a training dataset that needs to be split into train and validation sets.
    """
    
    def __init__(
        self,
        parquet_files: Union[str, Path],
        split: str,
        transform: Optional[Callable] = None
    ):
        """
        Args:
            parquet_files: Path to directory containing parquet shard files (part_*.parquet)
            split: Split to use ('train', 'val', or 'test')
            transform: Optional torchvision transform to apply to images
        """
        # Find all parquet files in the directory
        parquet_path = Path(parquet_files)
        all_parquet_files = sorted(parquet_path.glob("part_*.parquet"))
        #all_parquet_files = all_parquet_files[:10]
        
        if not all_parquet_files:
            raise FileNotFoundError(f"No parquet files found in directory: {parquet_path}")
        
        # Apply split logic: 50% train, 50% val
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got '{split}'")
        
        total_shards = len(all_parquet_files)
        split_point = int(total_shards * 0.8)
        if split == 'train':
            selected_parquet_files = all_parquet_files[:split_point]
            print(f"Using {len(selected_parquet_files)}/{total_shards} shards for training (50% split)")
        else:  # val or test
            selected_parquet_files = all_parquet_files[split_point:]
            print(f"Using {len(selected_parquet_files)}/{total_shards} shards for {split} (50% split)")
        
        # Pass the selected parquet files to the base class
        super().__init__(selected_parquet_files, transform)


class Yfcc4kPredictionDataset(PredictionBaseDataset):
    """
    Prediction dataset for YFCC4K.
    
    YFCC4K is a test/validation dataset only - no train/val split needed.
    """
    
    def __init__(
        self,
        parquet_files: Union[str, Path],
        split: str,
        transform: Optional[Callable] = None
    ):
        """
        Args:
            parquet_files: Path to directory containing parquet shard files (part_*.parquet)
            split: Split to use ('val' or 'test')
            transform: Optional torchvision transform to apply to images
        """
        if split not in ['val', 'test']:
            raise ValueError(f"Split must be 'val' or 'test', got '{split}'")
        
        # Find all parquet files in the directory
        parquet_path = Path(parquet_files)
        all_parquet_files = sorted(parquet_path.glob("part_*.parquet"))
        
        if not all_parquet_files:
            raise FileNotFoundError(f"No parquet files found in directory: {parquet_path}")
        
        # Pass all parquet files to the base class
        super().__init__(all_parquet_files, transform)


class Im2gps3kPredictionDataset(PredictionBaseDataset):
    """
    Prediction dataset for Im2GPS3K.
    
    Im2GPS3K is a test/validation dataset only - no train/val split needed.
    """
    
    def __init__(
        self,
        parquet_files: Union[str, Path],
        split: str,
        transform: Optional[Callable] = None
    ):
        """
        Args:
            parquet_files: Path to directory containing parquet shard files (part_*.parquet)
            split: Split to use ('val' or 'test')
            transform: Optional torchvision transform to apply to images
        """
        if split not in ['val', 'test']:
            raise ValueError(f"Split must be 'val' or 'test', got '{split}'")
        
        # Find all parquet files in the directory
        parquet_path = Path(parquet_files)
        all_parquet_files = sorted(parquet_path.glob("part_*.parquet"))
        
        if not all_parquet_files:
            raise FileNotFoundError(f"No parquet files found in directory: {parquet_path}")
        
        # Pass all parquet files to the base class
        super().__init__(all_parquet_files, transform)


