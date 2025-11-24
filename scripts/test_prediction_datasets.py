from data.datasets import load_prediction_dataset
from torch.utils.data import DataLoader
import torch


dataset_tests = {
    "mp16": ["train", "val"],
    "yfcc4k": [None],  # No split needed for test datasets
    "im2gps3k": [None],  # No split needed for test datasets
}

pred_model_name = "geoclip"


def test_dataset(dataset_name: str, split: str = None):
    """Test loading a dataset with a specific split."""
    print(f"\n{'='*60}")
    print(f"Testing {dataset_name}" + (f" (split: {split})" if split else " (no split)"))
    print(f"{'='*60}")
    
    try:
        # Load dataset
        dataset = load_prediction_dataset(
            dataset_name=dataset_name,
            model_name=pred_model_name,
            split=split
        )
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Dataset size: {len(dataset)}")
        
        # Test getting a single sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  Sample keys: {list(sample.keys())}")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: tensor shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"    {key}: {type(value).__name__}")
        
        # Test dataloader
        batch_size = 16
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
            # Only test first batch
            break
        
        print(f"✓ All tests passed for {dataset_name}" + (f" (split: {split})" if split else ""))
        return True
        
    except FileNotFoundError as e:
        print(f"✗ Dataset files not found: {e}")
        print(f"  Make sure prediction files exist for {dataset_name}_{pred_model_name}")
        return False
    except Exception as e:
        print(f"✗ Error testing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run tests for all datasets and splits."""
    results = {}
    
    for dataset_name, splits in dataset_tests.items():
        results[dataset_name] = {}
        for split in splits:
            success = test_dataset(dataset_name, split)
            results[dataset_name][split] = success
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for dataset_name, split_results in results.items():
        for split, success in split_results.items():
            status = "✓ PASS" if success else "✗ FAIL"
            split_str = f"split={split}" if split else "no split"
            print(f"{status} - {dataset_name} ({split_str})")
    
    # Check if all tests passed
    all_passed = all(
        success 
        for split_results in results.values() 
        for success in split_results.values()
    )
    
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed. Check the output above for details.")
    
    return all_passed


if __name__ == "__main__":
    main()
