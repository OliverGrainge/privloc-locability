from data.generate.yfcc100m import YFCC100MDataset

def load_dataset(dataset_name: str, **kwargs): 
    if dataset_name == "yfcc100m":
        return YFCC100MDataset(**kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")