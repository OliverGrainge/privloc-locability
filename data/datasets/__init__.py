from typing import Optional
from data.datasets.dataset import MP16Dataset, YFCC4kDataset, Im2GPS3kDataset
from data.datasets.predictiondataset import (
    PredictionBaseDataset,
    Mp16PredictionDataset,
    Yfcc4kPredictionDataset,
    Im2gps3kPredictionDataset,
    # Backward compatibility
    PredictionDataset,
    PredictionGeoDataset
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

def load_prediction_dataset(dataset_name: str, model_name: str, output_dir: str = None, split: Optional[str] = None, **kwargs):
    """
    Load a prediction dataset from parquet files (single file or sharded files).
    
    Args:
        dataset_name: Name of the dataset (e.g., 'mp16', 'yfcc4k', 'im2gps3k')
        model_name: Name of the model used for predictions (e.g., 'geoclip')
        output_dir: Directory containing prediction files (defaults to 'predictions')
        split: Split to use ('train' or 'val'). Only used for mp16 dataset.
        **kwargs: Additional arguments passed to the dataset class (e.g., transform, train_ratio for mp16)
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
    
    # Check if the sharded directory exists, otherwise fall back to single file
    if not parquet_path.is_dir():
        # Fall back to single file pattern for backward compatibility
        single_file_path = Path(output_dir) / f"{dataset_name}_{model_name}.parquet"
        if single_file_path.is_file():
            parquet_path = single_file_path
        else:
            raise FileNotFoundError(f"Neither directory {parquet_path} nor file {single_file_path} found")
    
    # Select the appropriate dataset class based on dataset_name
    if dataset_name == "mp16":
        return Mp16PredictionDataset(parquet_path, split=split, **kwargs)
    elif dataset_name == "yfcc4k":
        return Yfcc4kPredictionDataset(parquet_path, **kwargs)
    elif dataset_name == "im2gps3k":
        return Im2gps3kPredictionDataset(parquet_path, **kwargs)
    else:
        raise ValueError(f"Unknown prediction dataset: {dataset_name}")
