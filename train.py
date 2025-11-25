#!/usr/bin/env python3
"""
Training script for BinaryClassifier.

This script loads a configuration file and trains the binary error classification model
with checkpointing and logging capabilities.
"""
from dotenv import load_dotenv 
load_dotenv()

import argparse
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, List

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from models.binaryclassifer.model import BinaryClassifier
from data.datasets import load_prediction_dataset, prediction_collate_fn


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


def setup_checkpoint_callback(config: Dict[str, Any], config_dir: Path) -> ModelCheckpoint:
    """
    Setup model checkpoint callback.
    
    Args:
        config: Configuration dictionary
        config_dir: Directory containing the config file
        
    Returns:
        ModelCheckpoint callback or None if disabled
    """
    if not config.get('checkpoint', False):
        return None
    
    checkpoint_dir = config_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename='best_model',
        monitor='val/loss',
        mode='min',
        save_top_k=1,
        save_last=False,
        verbose=True
    )
    
    return checkpoint_callback


def setup_logger(config: Dict[str, Any], config_dir: Path, config_name: str) -> WandbLogger:
    """
    Setup Weights & Biases logger.
    
    Args:
        config: Configuration dictionary
        config_dir: Directory containing the config file
        config_name: Name of the config file (without extension)
        
    Returns:
        WandbLogger or None if logging is disabled
    """
    if not config.get('logger', False):
        return None
    
    logs_dir = config_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    logger = WandbLogger(
        project="GeoErrorBinaryClassifier",
        name=f"{config_name}_training",
        save_dir=str(logs_dir),
        log_model=False
    )
    
    return logger



def create_model(config: Dict[str, Any]) -> BinaryClassifier:
    """
    Create the binary error classifier from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized BinaryClassifier
    """
    model_config = config['model']
    model = BinaryClassifier(**model_config)
    return model


def create_dataloaders(model: BinaryClassifier, config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, List[DataLoader]]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        model: The BinaryClassifier model (to access transform)
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loaders_list)
    """
    data_config = config['data']
    
    # Get train config
    train_config = data_config['train']
    train_dataset_name = train_config.get('dataset_name', 'mp16')  # Default to mp16 if not specified
    train_pred_model_name = train_config['pred_model_name']
    train_batch_size = train_config['batch_size']
    train_num_workers = train_config['num_workers']
    
    # Create training dataloader
    train_dataset = load_prediction_dataset(
        dataset_name=train_dataset_name,
        model_name=train_pred_model_name,
        transform=model.transform,
        split="train",
    )
    drop_last_train = len(train_dataset) > train_batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=train_num_workers,
        pin_memory=True,
        drop_last=drop_last_train,
        collate_fn=prediction_collate_fn,
    )
    
    # Create validation dataloader (use same config as train)
    val_dataset = load_prediction_dataset(
        dataset_name=train_dataset_name,
        model_name=train_pred_model_name,
        transform=model.transform,
        split="val",
    )
    drop_last_val = len(val_dataset) > train_batch_size
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=train_num_workers,
        pin_memory=True,
        drop_last=drop_last_val,
        collate_fn=prediction_collate_fn,
    )
    
    # Create test dataloaders from test config list
    test_loaders = []
    test_configs = data_config.get('test', [])
    
    for test_item in test_configs:
        if isinstance(test_item, dict):
            # Get dataset name and config
            dataset_name = test_item['dataset_name']
            test_pred_model_name = test_item['pred_model_name']
            test_batch_size = test_item['batch_size']
            test_num_workers = test_item['num_workers']
            
            # For test datasets, use "test" split
            test_dataset = load_prediction_dataset(
                dataset_name=dataset_name,
                model_name=test_pred_model_name,
                transform=model.transform,
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
            test_loaders.append(test_loader)
    
    return train_loader, val_loader, test_loaders


def setup_trainer(config: Dict[str, Any], config_dir: Path, config_name: str) -> pl.Trainer:
    """
    Setup PyTorch Lightning trainer.
    
    Args:
        config: Configuration dictionary
        config_dir: Directory containing the config file
        config_name: Name of the config file (without extension)
        
    Returns:
        Configured PyTorch Lightning trainer
    """
    trainer_config = config['trainer']
    
    # Setup callbacks
    callbacks = []
    
    # Add checkpoint callback if enabled
    checkpoint_callback = setup_checkpoint_callback(config, config_dir)
    if checkpoint_callback:
        callbacks.append(checkpoint_callback)
    
    
    # Add learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    
    # Setup logger
    logger = setup_logger(config, config_dir, config_name)
    
    # Create trainer with all configuration parameters
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        **trainer_config
    )
    
    return trainer


def main():
    """
    Main training function.
    """
    parser = argparse.ArgumentParser(description='Train BinaryClassifier')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to the configuration YAML file'
    )
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file to resume training from'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract config name and directory from path
    config_path = Path(args.config)
    config_name = config_path.stem
    config_dir = config_path.parent
    
    print(f"Training with configuration: {config_name}")
    print(f"Config file: {args.config}")
    print(f"Config directory: {config_dir}")
    print(f"Checkpoints will be saved to: {config_dir / 'checkpoints'}")
    print(f"Logs will be saved to: {config_dir / 'logs'}")
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    
    # Setup trainer
    print("Setting up trainer...")
    trainer = setup_trainer(config, config_dir, config_name)
    
    # Print model summary
    print("\nModel Summary:")
    print(f"Architecture: {config['model']['arch_name']}")
    print(f"Threshold: {config['model']['threshold_km']} km")
    print(f"Learning Rate: {config['model']['learning_rate']}")
    
    # Print data summary
    print("\nData Configuration:")
    train_config = config['data']['train']
    print(f"Training Dataset: {train_config.get('dataset_name', 'mp16')}")
    print(f"Training Prediction Model: {train_config['pred_model_name']}")
    print(f"Training Batch Size: {train_config['batch_size']}")
    print(f"Test Datasets: {len(config['data'].get('test', []))}")
    print(f"\nTraining Configuration:")
    print(f"Max Steps: {config['trainer'].get('max_steps', 'N/A')}")
    print(f"Max Epochs: {config['trainer'].get('max_epochs', 'N/A')}")
    print(f"Validation Check Interval: {config['trainer'].get('val_check_interval', config['trainer'].get('check_val_every_n_epoch', 'N/A'))} {'steps' if 'val_check_interval' in config['trainer'] else 'epochs'}")
    print(f"Max Validation Batches: {config['trainer'].get('limit_val_batches', 'All')}")
    print(f"Max Test Batches: {config['trainer'].get('limit_test_batches', 'All')}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loaders = create_dataloaders(model, config)
    
    # Check dataset sizes
    print("\nChecking dataset sizes...")
    print(f"Training dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    for i, test_loader in enumerate(test_loaders):
        print(f"Test dataset {i+1} size: {len(test_loader.dataset)}")
        print(f"Test dataset {i+1} batches: {len(test_loader)}")
    
    if len(train_loader) == 0:
        train_config = config['data']['train']
        raise ValueError("Training dataloader is empty! Check your dataset configuration. "
                        f"Dataset has {len(train_loader.dataset)} samples but batch_size is {train_config['batch_size']}. "
                        "Consider reducing batch_size or checking if drop_last=True is causing issues.")
    
    # Start training
    print("\nStarting training...")

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from_checkpoint if args.resume_from_checkpoint else None
    )
    print("Training completed successfully!")
    
    # Print best model path if checkpointing was enabled
    best_model_path = None
    if config.get('checkpoint', False):
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                best_model_path = callback.best_model_path
                print(f"Best model saved at: {best_model_path}")
                break
    
    print("\nTraining completed! Use test.py to run evaluation on test datasets.")
            



if __name__ == "__main__":
    main()