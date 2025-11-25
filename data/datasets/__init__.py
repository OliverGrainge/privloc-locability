from typing import Optional, Callable
from data.datasets.dataset import MP16Dataset, YFCC4kDataset, Im2GPS3kDataset
from data.datasets.predictiondataset import (
    Mp16PredictionDataset,
    Yfcc4kPredictionDataset,
    Im2gps3kPredictionDataset,
    prediction_collate_fn,
)

def load_dataset(dataset_name: str, **kwargs): 
    if dataset_name == "mp16":
        return MP16Dataset(**kwargs)
    elif dataset_name == "yfcc4k":
        return YFCC4kDataset(**kwargs)
    elif dataset_name == "im2gps3k":
        return Im2GPS3kDataset(**kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

def load_prediction_dataset(dataset_name: str, model_name: str, output_dir: str = None, split: Optional[str] = None, transform: Optional[Callable] = None):
    """
    Load a prediction dataset from parquet files (sharded files).
    
    Args:
        dataset_name: Name of the dataset (e.g., 'mp16', 'yfcc4k', 'im2gps3k')
        model_name: Name of the model used for predictions (e.g., 'geoclip')
        output_dir: Directory containing prediction files (defaults to 'predictions')
        split: Split to use ('train', 'val', or 'test' for mp16; 'val' or 'test' for yfcc4k/im2gps3k)
        transform: Optional torchvision transform to apply to images
    """
    import os
    from pathlib import Path
    
    # Use DATASET_DIR environment variable if available, otherwise use default
    if output_dir is None:
        dataset_dir = os.getenv('DATASET_DIR')
        if dataset_dir:
            output_dir = os.path.join(dataset_dir, 'predictions')
        else:
            output_dir = 'predictions'
    
    # Construct parquet path following the same pattern as generate.py
    # generate.py creates sharded files in: {output_dir}/{dataset_name}_{model_name}/part_*.parquet
    parquet_path = Path(output_dir) / f"{dataset_name}_{model_name}"
    
    if not parquet_path.is_dir():
        raise FileNotFoundError(f"Parquet directory not found: {parquet_path}")
    
    # Select the appropriate dataset class based on dataset_name
    if dataset_name == "mp16":
        if split is None:
            raise ValueError("split must be provided for mp16 dataset ('train', 'val', or 'test')")
        return Mp16PredictionDataset(parquet_path, split, transform)
    elif dataset_name == "yfcc4k":
        if split is None:
            raise ValueError("split must be provided for yfcc4k dataset ('val' or 'test')")
        return Yfcc4kPredictionDataset(parquet_path, split, transform)
    elif dataset_name == "im2gps3k":
        if split is None:
            raise ValueError("split must be provided for im2gps3k dataset ('val' or 'test')")
        return Im2gps3kPredictionDataset(parquet_path, split, transform)
    else:
        raise ValueError(f"Unknown prediction dataset: {dataset_name}")
