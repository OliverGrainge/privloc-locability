#!/usr/bin/env python3
"""
Testing script for BinaryClassifier.

This script loads a trained model and runs evaluation on test datasets,
saving results to CSV files.
"""
from dotenv import load_dotenv 
load_dotenv()

import argparse
import os
import yaml
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models import BinaryClassifier, ConditionalVAEClassifier
from data.datasets import load_prediction_dataset, prediction_collate_fn


MODEL_REGISTRY = {
    "binary_classifier": BinaryClassifier,
    "conditional_vae": ConditionalVAEClassifier,
}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_model_config(model_config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Extract model type and parameters from configuration.
    """
    if 'type' not in model_config:
        raise ValueError("Model configuration must include a 'type' field.")

    model_type = model_config['type'].lower()
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'. "
                         f"Available types: {sorted(MODEL_REGISTRY.keys())}")

    if 'params' in model_config:
        model_params = model_config['params']
    else:
        model_params = {k: v for k, v in model_config.items() if k != 'type'}

    if not isinstance(model_params, dict):
        raise ValueError("Model 'params' must be a dictionary of keyword arguments.")

    return model_type, model_params


def create_test_dataloaders(model: pl.LightningModule, config: Dict[str, Any]) -> List[DataLoader]:
    """
    Create test dataloaders from config.
    
    Args:
        model: The BinaryClassifier model (to access transform)
        config: Configuration dictionary
        
    Returns:
        List of test dataloaders
    """
    data_config = config['data']
    test_loaders = []
    test_configs = data_config.get('test', [])
    
    for test_item in test_configs:
        if isinstance(test_item, dict):
            dataset_name = test_item['dataset_name']
            test_pred_model_name = test_item['pred_model_name']
            test_batch_size = test_item['batch_size']
            test_num_workers = test_item['num_workers']
            
            # For test datasets, use "test" split
            test_dataset = load_prediction_dataset(
                dataset_name=dataset_name,
                model_name=test_pred_model_name,
                transform=getattr(model, "transform", None),
                split="test",
            )
            drop_last_test = len(test_dataset) > test_batch_size
            test_loader = DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                shuffle=False,
                num_workers=test_num_workers,
                pin_memory=True,
                drop_last=drop_last_test,
                collate_fn=prediction_collate_fn,
            )
            test_loaders.append((dataset_name, test_loader))
    
    return test_loaders


def extract_metrics_from_trainer(trainer: pl.Trainer, model: pl.LightningModule, dataset_name: str) -> Dict[str, Any]:
    """
    Extract metrics from trainer and model.
    
    Args:
        trainer: PyTorch Lightning trainer
        model: The BinaryClassifier model
        dataset_name: Name of the test dataset
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    def extract_value(value):
        """Extract float value from tensor or other types."""
        if hasattr(value, 'item'):
            return float(value.item())
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return value
    
    # Get all logged metrics from trainer
    if hasattr(trainer, 'logged_metrics'):
        for key, value in trainer.logged_metrics.items():
            if key.startswith('test/'):
                metric_name = key.replace('test/', '')
                metrics[f"{dataset_name}_{metric_name}"] = extract_value(value)
    
    # Also check callback_metrics
    if hasattr(trainer, 'callback_metrics'):
        for key, value in trainer.callback_metrics.items():
            if key.startswith('test/'):
                metric_name = key.replace('test/', '')
                # Only add if not already in logged_metrics
                full_key = f"{dataset_name}_{metric_name}"
                if full_key not in metrics:
                    metrics[full_key] = extract_value(value)
    
    # Also check progress_bar_metrics
    if hasattr(trainer, 'progress_bar_metrics'):
        for key, value in trainer.progress_bar_metrics.items():
            if key.startswith('test/'):
                metric_name = key.replace('test/', '')
                full_key = f"{dataset_name}_{metric_name}"
                if full_key not in metrics:
                    metrics[full_key] = extract_value(value)
    
    return metrics


def save_results_to_csv(results: List[Dict[str, Any]], config_dir: Path, config_name: str, timestamp: str):
    """
    Save test results to CSV file.
    
    Args:
        results: List of result dictionaries (one per test dataset)
        config_dir: Directory containing the config file
        config_name: Name of the config file (without extension)
        timestamp: Timestamp string for folder name
    """
    results_dir = config_dir / "results" / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename without timestamp (since it's in timestamped folder)
    csv_path = results_dir / f"{config_name}_test_results.csv"
    
    # Collect all unique keys from all results
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())
    all_keys = sorted(all_keys)
    
    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {csv_path}")
    return csv_path


def save_results_to_txt(results: List[Dict[str, Any]], config_dir: Path, config_name: str, csv_path: Path, timestamp: str):
    """
    Save test results to a nicely formatted text file.
    
    Args:
        results: List of result dictionaries (one per test dataset)
        config_dir: Directory containing the config file
        config_name: Name of the config file (without extension)
        csv_path: Path to the CSV file (for reference)
        timestamp: Timestamp string for folder name
    """
    results_dir = config_dir / "results" / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename without timestamp (since it's in timestamped folder)
    txt_path = results_dir / f"{config_name}_test_results.txt"
    
    # Collect all metric keys (exclude metadata keys)
    metadata_keys = {'dataset_name', 'checkpoint', 'config', 'timestamp'}
    
    # Extract base metric names by removing dataset prefixes
    # Metrics are stored as "{dataset_name}_{metric_name}"
    base_metric_names = set()
    for result in results:
        dataset_name = result.get('dataset_name', '')
        for key in result.keys():
            if key not in metadata_keys:
                # Check if key starts with dataset name prefix
                if key.startswith(f"{dataset_name}_"):
                    base_name = key[len(f"{dataset_name}_"):]
                    base_metric_names.add(base_name)
                else:
                    # If no prefix, use the key as-is
                    base_metric_names.add(key)
    
    base_metric_names = sorted(base_metric_names)
    
    # Determine column widths
    dataset_col_width = max(len('Dataset'), max(len(r.get('dataset_name', '')) for r in results))
    metric_col_widths = {metric: max(len(metric), 10) for metric in base_metric_names}
    
    # Build the table
    lines = []
    lines.append("=" * 80)
    lines.append("TEST RESULTS")
    lines.append("=" * 80)
    lines.append(f"Configuration: {config_name}")
    lines.append(f"Timestamp: {results[0].get('timestamp', 'N/A')}")
    lines.append(f"Checkpoint: {results[0].get('checkpoint', 'N/A')}")
    lines.append("")
    
    # Header row
    header = f"{'Dataset':<{dataset_col_width}}"
    for metric in base_metric_names:
        header += f" | {metric:<{metric_col_widths[metric]}}"
    lines.append(header)
    lines.append("-" * len(header))
    
    # Data rows
    for result in results:
        dataset_name = result.get('dataset_name', 'N/A')
        row = f"{dataset_name:<{dataset_col_width}}"
        for metric in base_metric_names:
            # Look for metric with dataset prefix
            prefixed_key = f"{dataset_name}_{metric}"
            value = result.get(prefixed_key, 'N/A')
            
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            elif isinstance(value, (int, float)):
                formatted_value = str(value)
            else:
                formatted_value = str(value)
            row += f" | {formatted_value:<{metric_col_widths[metric]}}"
        lines.append(row)
    
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"Detailed results are also available in CSV format:")
    lines.append(f"  {csv_path}")
    lines.append("=" * 80)
    
    # Write to file
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Results saved to: {txt_path}")
    return txt_path


def main():
    """
    Main testing function.
    """
    parser = argparse.ArgumentParser(description='Test BinaryClassifier')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to the configuration YAML file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file to load model from (optional, will auto-detect from config dir if not provided)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract config name and directory from path
    config_path = Path(args.config)
    config_name = config_path.stem
    config_dir = config_path.parent
    
    # Find checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        # Auto-detect checkpoint from config directory
        checkpoint_dir = config_dir / "checkpoints"
        
        # First, try to find best_model.ckpt (the standard name)
        best_model_path = checkpoint_dir / "best_model.ckpt"
        
        if best_model_path.exists():
            checkpoint_path = best_model_path
        else:
            # Look for any .ckpt file in the checkpoint directory
            ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
            if len(ckpt_files) == 0:
                raise FileNotFoundError(
                    f"No checkpoint found in {checkpoint_dir}. "
                    f"Either provide --checkpoint argument or ensure checkpoints exist in the config directory."
                )
            elif len(ckpt_files) == 1:
                checkpoint_path = ckpt_files[0]
            else:
                # If multiple checkpoints, use the most recently modified one
                checkpoint_path = max(ckpt_files, key=lambda p: p.stat().st_mtime)
                print(f"Warning: Multiple checkpoints found. Using most recent: {checkpoint_path.name}")
    
    print(f"Testing with configuration: {config_name}")
    print(f"Config file: {args.config}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Results will be saved to: {config_dir / 'results'}")
    
    # Create model
    print("\nLoading model from checkpoint...")
    model_type, model_params = parse_model_config(config['model'])
    model_cls = MODEL_REGISTRY[model_type]
    model = model_cls.load_from_checkpoint(
        str(checkpoint_path),
        strict=False,
        **model_params,
    )
    
    # Create test dataloaders
    print("\nCreating test dataloaders...")
    test_loaders = create_test_dataloaders(model, config)
    
    if len(test_loaders) == 0:
        raise ValueError("No test datasets configured in config file")
    
    print(f"Found {len(test_loaders)} test dataset(s)")
    for dataset_name, loader in test_loaders:
        print(f"  - {dataset_name}: {len(loader.dataset)} samples, {len(loader)} batches")
    
    # Setup trainer for testing
    trainer_config = config.get('trainer', {})
    logger = None
    if config.get('logger', False):
        logs_dir = config_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        logger = WandbLogger(
            project="GeoErrorBinaryClassifier",
            name=f"{config_name}_testing",
            save_dir=str(logs_dir),
            log_model=False
        )
    
    trainer = pl.Trainer(
        logger=logger,
        **trainer_config
    )
    
    # Run testing on all datasets
    print("\nRunning test evaluation...")
    all_results = []
    
    # Create a single timestamp for this test run (all files will go in the same folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = config_dir / "results" / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset_name, test_loader in test_loaders:
        print(f"\n{'='*60}")
        print(f"Testing on dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Run test
        trainer.test(model, dataloaders=test_loader)
        
        # Save ROC curve figure if it exists
        if hasattr(trainer, 'test_roc_figure') and trainer.test_roc_figure is not None:
            roc_path = results_dir / f"{dataset_name}_test_roc_curve.png"
            trainer.test_roc_figure.savefig(roc_path, dpi=150, bbox_inches='tight')
            print(f"ROC curve saved to: {roc_path}")
            # Clean up the figure
            plt.close(trainer.test_roc_figure)
            trainer.test_roc_figure = None
        
        # Save probability distribution figure if it exists
        if hasattr(trainer, 'test_prob_dist_figure') and trainer.test_prob_dist_figure is not None:
            dist_path = results_dir / f"{dataset_name}_test_prob_distribution.png"
            trainer.test_prob_dist_figure.savefig(dist_path, dpi=150, bbox_inches='tight')
            print(f"Probability distribution saved to: {dist_path}")
            plt.close(trainer.test_prob_dist_figure)
            trainer.test_prob_dist_figure = None
        
        # Extract metrics
        metrics = extract_metrics_from_trainer(trainer, model, dataset_name)
        metrics['dataset_name'] = dataset_name
        metrics['checkpoint'] = str(checkpoint_path)
        metrics['config'] = args.config
        metrics['timestamp'] = datetime.now().isoformat()
        
        all_results.append(metrics)
        
        # Print summary
        print(f"\nResults for {dataset_name}:")
        for key, value in metrics.items():
            if key not in ['dataset_name', 'checkpoint', 'config', 'timestamp']:
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Save results to CSV and TXT
    print("\n" + "="*60)
    csv_path = save_results_to_csv(all_results, config_dir, config_name, timestamp)
    txt_path = save_results_to_txt(all_results, config_dir, config_name, csv_path, timestamp)
    print("="*60)
    print("Testing completed successfully!")
    print(f"Results saved to: {results_dir}")
    print(f"  CSV: {csv_path.name}")
    print(f"  TXT: {txt_path.name}")


if __name__ == "__main__":
    main()

