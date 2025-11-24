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


class PredictionBaseDataset(Dataset):
    """
    Base PyTorch Dataset for loading prediction results from parquet files.
    Supports both single parquet files and sharded parquet files.
    
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
        parquet_path: Union[str, Path],
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            parquet_path: Path to parquet file/directory containing prediction results.
                         Can be a single file or a directory containing sharded files (part_*.parquet)
            transform: Optional torchvision transform to apply to images
        """
        self.parquet_path = Path(parquet_path)
        self.transform = transform
        
        # Determine if we have a single file or sharded files
        if self.parquet_path.is_file():
            # Single parquet file
            self.parquet_files = [self.parquet_path]
        elif self.parquet_path.is_dir():
            # Directory with sharded files
            self.parquet_files = sorted(self.parquet_path.glob("part_*.parquet"))
            if not self.parquet_files:
                raise FileNotFoundError(f"No parquet files found in directory: {self.parquet_path}")
        else:
            raise FileNotFoundError(f"Parquet path not found: {self.parquet_path}")

        # Initialize lazy loading state
        self.current_file_idx = None
        self.current_df = None
        self.file_start_indices = []
        self.total_samples = 0
        
        # Prefetching state
        self.prefetch_next = True
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
            
            # Apply transform if provided, otherwise return PIL image
            if self.transform:
                image = self.transform(pil_image)
            else:
                image = pil_image
            
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
        
        # Apply transform if provided, otherwise return PIL image
        if self.transform:
            image = self.transform(pil_image)
        else:
            image = pil_image
        
        result = {
            'image': image,
            'pil_image': pil_image,  # Always include PIL image for plonk
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
            'parquet_path': str(self.parquet_path),
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
        parquet_path: Union[str, Path],
        transform: Optional[Callable] = None,
        split: Optional[str] = None,
        train_ratio: float = 0.9,
    ):
        """
        Args:
            parquet_path: Path to parquet file/directory containing prediction results
            transform: Optional torchvision transform to apply to images
            split: Split to use ('train' or 'val'). If None, uses all data.
            train_ratio: Ratio of data to use for training (default 0.9 = 90% train, 10% val)
        """
        # First, initialize the base dataset to get all files
        super().__init__(parquet_path, transform)
        
        self.split = split
        self.train_ratio = train_ratio
        
        # Apply split logic if specified
        self.split_start_idx = None
        self.split_end_idx = None
        
        if self.split is not None:
            if self.split not in ['train', 'val']:
                raise ValueError(f"Split must be 'train' or 'val', got '{self.split}'")
            
            total_shards = len(self.parquet_files)
            
            # If only 1 shard, partition data within the shard
            if total_shards == 1:
                # Get the total number of rows in the single shard
                df_meta = pd.read_parquet(self.parquet_files[0], engine='pyarrow')
                total_rows = len(df_meta)
                train_rows = int(total_rows * self.train_ratio)
                
                if self.split == 'train':
                    self.split_start_idx = 0
                    self.split_end_idx = train_rows
                    self.total_samples = train_rows
                    print(f"Partitioning single shard: using rows 0-{train_rows} for training (ratio={self.train_ratio}, {train_rows}/{total_rows} rows)")
                else:  # val
                    self.split_start_idx = train_rows
                    self.split_end_idx = total_rows
                    self.total_samples = total_rows - train_rows
                    print(f"Partitioning single shard: using rows {train_rows}-{total_rows} for validation (ratio={self.train_ratio}, {self.total_samples}/{total_rows} rows)")
            else:
                # Multiple shards: split by shards
                train_shards = int(total_shards * self.train_ratio)
                
                if self.split == 'train':
                    self.parquet_files = self.parquet_files[:train_shards]
                    # Recalculate total_samples and file_start_indices
                    self._recalculate_indices()
                    print(f"Using {len(self.parquet_files)} shards for training (ratio={self.train_ratio})")
                else:  # val
                    self.parquet_files = self.parquet_files[train_shards:]
                    # Recalculate total_samples and file_start_indices
                    self._recalculate_indices()
                    print(f"Using {len(self.parquet_files)} shards for validation (ratio={self.train_ratio})")
    
    def _recalculate_indices(self):
        """Recalculate file_start_indices and total_samples after filtering parquet_files."""
        self.file_start_indices = []
        self.total_samples = 0
        
        for i, file_path in enumerate(self.parquet_files):
            df_meta = pd.read_parquet(file_path, engine='pyarrow')
            file_rows = len(df_meta)
            self.file_start_indices.append(self.total_samples)
            self.total_samples += file_rows
    
    def __getitem__(self, idx: int) -> dict:
        """Override to handle single-shard split."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        # If we're splitting within a single shard, adjust the index
        if self.split_start_idx is not None:
            # Find which file contains this index (should be file 0 for single shard split)
            file_idx = 0
            self._load_file_if_needed(file_idx)
            
            # Calculate the local index within the split portion
            local_idx = self.split_start_idx + idx
            
            # Get the row data
            row = self.current_df.iloc[local_idx]
        else:
            # Use parent implementation for multi-shard splits
            return super().__getitem__(idx)
        
        # Decode image from bytes
        image_bytes = row['image_bytes']
        if isinstance(image_bytes, str):
            image_bytes = image_bytes.encode('latin1')
        
        pil_image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Apply transform if provided, otherwise return PIL image
        if self.transform:
            image = self.transform(pil_image)
        else:
            image = pil_image
        
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
            import numpy as np
            embedding_bytes = row['embedding']
            if isinstance(embedding_bytes, bytes):
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                result['embedding'] = torch.tensor(embedding)
        
        return result
    
    def get_file_samples(self, file_idx: int) -> List[dict]:
        """Override to handle single-shard split."""
        if file_idx < 0 or file_idx >= len(self.parquet_files):
            raise IndexError(f"File index {file_idx} out of range for {len(self.parquet_files)} files")
        
        self._load_file_if_needed(file_idx)
        
        samples = []
        # Determine the range of rows to iterate over
        if self.split_start_idx is not None and file_idx == 0:
            # Single shard split: only iterate over the split portion
            start_row = self.split_start_idx
            end_row = self.split_end_idx
        else:
            start_row = 0
            end_row = len(self.current_df)
        
        for i in range(start_row, end_row):
            # Calculate global index (relative to the split)
            local_idx_in_split = i - start_row
            global_idx = local_idx_in_split
                
            row = self.current_df.iloc[i]
            
            # Decode image from bytes
            image_bytes = row['image_bytes']
            if isinstance(image_bytes, str):
                image_bytes = image_bytes.encode('latin1')
            
            pil_image = Image.open(BytesIO(image_bytes)).convert('RGB')
            
            # Apply transform if provided, otherwise return PIL image
            if self.transform:
                image = self.transform(pil_image)
            else:
                image = pil_image
            
            sample = {
                'image': image,
                'pil_image': pil_image,
                'true_lat': torch.tensor(float(row['true_lat'])),
                'true_lon': torch.tensor(float(row['true_lon'])),
                'pred_lat': torch.tensor(float(row['pred_lat'])),
                'pred_lon': torch.tensor(float(row['pred_lon'])),
                'idx': torch.tensor(global_idx)
            }
            
            # Add embedding if available
            if self.has_embeddings and 'embedding' in row:
                import numpy as np
                embedding_bytes = row['embedding']
                if isinstance(embedding_bytes, bytes):
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    sample['embedding'] = torch.tensor(embedding)
            
            samples.append(sample)
        
        return samples


class Yfcc4kPredictionDataset(PredictionBaseDataset):
    """
    Prediction dataset for YFCC4K.
    
    YFCC4K is a test/validation dataset only - no train/val split needed.
    """
    pass  # No additional logic needed, just use the base class


class Im2gps3kPredictionDataset(PredictionBaseDataset):
    """
    Prediction dataset for Im2GPS3K.
    
    Im2GPS3K is a test/validation dataset only - no train/val split needed.
    """
    pass  # No additional logic needed, just use the base class


# Backward compatibility alias
PredictionDataset = PredictionBaseDataset
PredictionGeoDataset = PredictionBaseDataset
