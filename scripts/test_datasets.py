from dotenv import load_dotenv 
load_dotenv()

from data.datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch


# Test configurations for each dataset
dataset_configs = {
    "mp16": {
        "kwargs": {
            "transform": None,  # Will test with and without transform
            "max_shards": 2,  # Limit shards for faster testing
        }
    },
    "yfcc4k": {
        "kwargs": {
            "transform": None,
        }
    },
    "im2gps3k": {
        "kwargs": {
            "transform": None,
        }
    },
}

# Simple transform for testing
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def test_dataset(dataset_name: str, use_transform: bool = False):
    """Test loading a dataset with or without transform."""
    transform_label = "with transform" if use_transform else "without transform"
    print(f"\n{'='*60}")
    print(f"Testing {dataset_name} ({transform_label})")
    print(f"{'='*60}")
    
    try:
        # Get config for this dataset
        config = dataset_configs.get(dataset_name, {"kwargs": {}})
        kwargs = config["kwargs"].copy()
        
        # Apply transform if requested
        if use_transform:
            kwargs["transform"] = test_transform
        else:
            kwargs["transform"] = None
        
        # Load dataset
        dataset = load_dataset(dataset_name, **kwargs)
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Dataset size: {len(dataset)}")
        
        # Test getting a single sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  Sample keys: {list(sample.keys())}")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: tensor shape {value.shape}, dtype {value.dtype}")
                elif isinstance(value, bytes):
                    print(f"    {key}: bytes (length: {len(value)})")
                else:
                    print(f"    {key}: {type(value).__name__} = {value}")
            
            # Verify required keys exist
            required_keys = ['image', 'lat', 'lon', 'image_bytes']
            missing_keys = [key for key in required_keys if key not in sample]
            if missing_keys:
                print(f"  ⚠ Warning: Missing expected keys: {missing_keys}")
            else:
                print(f"  ✓ All required keys present")
        
        # Test dataloader (only if dataset has samples and transform is provided)
        # Note: DataLoader can't batch PIL Images with default collate, so we skip when transform=None
        if len(dataset) > 0:
            # Check if image is a PIL Image (no transform) or tensor (with transform)
            from PIL import Image as PILImage
            sample_image = sample.get('image')
            is_pil_image = isinstance(sample_image, PILImage.Image)
            
            if is_pil_image and not use_transform:
                print(f"\n  ⚠ Skipping DataLoader test (transform=None returns PIL Images)")
                print(f"    DataLoader's default collate cannot batch PIL Images.")
                print(f"    Use a transform (e.g., ToTensor()) to enable batching.")
            else:
                batch_size = min(8, len(dataset))  # Use smaller batch size for testing
                dataloader = DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    shuffle=False,
                    num_workers=0  # Use 0 for testing to avoid multiprocessing issues
                )
                
                print(f"\n  Testing DataLoader (batch_size={batch_size})...")
                for i, batch in enumerate(dataloader):
                    print(f"  ✓ Batch {i+1} loaded successfully")
                    print(f"    Batch keys: {list(batch.keys())}")
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            print(f"      {key}: tensor shape {value.shape}, dtype {value.dtype}")
                        elif isinstance(value, list):
                            print(f"      {key}: list (length: {len(value)})")
                            if len(value) > 0:
                                print(f"        First item type: {type(value[0]).__name__}")
                    # Only test first batch
                    break
                
                print(f"  ✓ DataLoader test passed")
        else:
            print(f"  ⚠ Skipping DataLoader test (dataset is empty)")
        
        print(f"✓ All tests passed for {dataset_name} ({transform_label})")
        return True
        
    except ValueError as e:
        if "not found" in str(e).lower() or "not exist" in str(e).lower():
            print(f"✗ Dataset files not found: {e}")
            print(f"  Make sure the dataset is properly downloaded and available")
        else:
            print(f"✗ ValueError: {e}")
        return False
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
        print(f"  Make sure the dataset files exist")
        return False
    except Exception as e:
        print(f"✗ Error testing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run tests for all datasets."""
    results = {}
    
    # Test each dataset with and without transforms
    for dataset_name in dataset_configs.keys():
        results[dataset_name] = {}
        
        # Test without transform
        success_no_transform = test_dataset(dataset_name, use_transform=False)
        results[dataset_name]["no_transform"] = success_no_transform
        
        # Test with transform
        success_with_transform = test_dataset(dataset_name, use_transform=True)
        results[dataset_name]["with_transform"] = success_with_transform
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for dataset_name, test_results in results.items():
        for test_type, success in test_results.items():
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"{status} - {dataset_name} ({test_type})")
    
    # Check if all tests passed
    all_passed = all(
        success 
        for test_results in results.values() 
        for success in test_results.values()
    )
    
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed. Check the output above for details.")
        print("\nNote: If tests failed due to missing files, make sure:")
        print("  - Datasets are downloaded (check KAGGLEHUB_CACHE environment variable)")
        print("  - Required environment variables are set")
        print("  - File paths are correct")
    
    return all_passed


if __name__ == "__main__":
    main()

