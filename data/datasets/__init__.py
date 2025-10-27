from data.datasets.geodataset import MP16Dataset
from data.datasets.predictiongeodataset import PredictionGeoDataset

def load_dataset(dataset_name: str, **kwargs): 
    if dataset_name == "mp16":
        return MP16Dataset(**kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

def load_prediction_dataset(dataset_name: str, model_name: str, output_dir: str = None, **kwargs):
    """
    Load a prediction dataset from parquet files (single file or sharded files).
    
    Args:
        dataset_name: Name of the dataset (e.g., 'mp16')
        model_name: Name of the model used for predictions (e.g., 'geoclip')
        output_dir: Directory containing prediction files (defaults to 'predictions')
        **kwargs: Additional arguments passed to PredictionGeoDataset
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
    if parquet_path.is_dir():
        # Use the directory containing sharded files
        return PredictionGeoDataset(parquet_path, **kwargs)
    else:
        # Fall back to single file pattern for backward compatibility
        single_file_path = Path(output_dir) / f"{dataset_name}_{model_name}.parquet"
        return PredictionGeoDataset(single_file_path, **kwargs)