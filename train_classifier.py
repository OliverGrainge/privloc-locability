#!/usr/bin/env python3
"""
Training script for BinaryErrorClassifier.

This script loads a configuration file and trains the binary error classification model
with checkpointing and logging capabilities.
"""
from dotenv import load_dotenv 
load_dotenv()

import argparse
import os
import yaml
from pathlib import Path
from typing import Dict, Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from models.predlocate import BinaryErrorClassifier


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


def setup_checkpoint_callback(config: Dict[str, Any], config_name: str) -> ModelCheckpoint:
    """
    Setup model checkpoint callback.
    
    Args:
        config: Configuration dictionary
        config_name: Name of the config file (without extension)
        
    Returns:
        ModelCheckpoint callback or None if disabled
    """
    if not config.get('checkpoint', False):
        return None
    
    checkpoint_dir = f"checkpoints/{config_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best_model-{epoch:02d}-{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    
    return checkpoint_callback


def setup_logger(config: Dict[str, Any], config_name: str) -> WandbLogger:
    """
    Setup Weights & Biases logger.
    
    Args:
        config: Configuration dictionary
        config_name: Name of the config file (without extension)
        
    Returns:
        WandbLogger or None if logging is disabled
    """
    if not config.get('logger', False):
        return None
    
    logger = WandbLogger(
        project="GeoErrorBinaryClassifier",
        name=f"{config_name}_training",
        save_dir="logs/",
        log_model=False
    )
    
    return logger



def create_model(config: Dict[str, Any]) -> BinaryErrorClassifier:
    """
    Create the binary error classifier from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized BinaryErrorClassifier
    """
    model_config = config['model']
    model = BinaryErrorClassifier(**model_config)
    return model


def setup_trainer(config: Dict[str, Any], config_name: str) -> pl.Trainer:
    """
    Setup PyTorch Lightning trainer.
    
    Args:
        config: Configuration dictionary
        config_name: Name of the config file (without extension)
        
    Returns:
        Configured PyTorch Lightning trainer
    """
    trainer_config = config['trainer']
    
    # Setup callbacks
    callbacks = []
    
    # Add checkpoint callback if enabled
    checkpoint_callback = setup_checkpoint_callback(config, config_name)
    if checkpoint_callback:
        callbacks.append(checkpoint_callback)
    
    
    # Add learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    
    # Setup logger
    logger = setup_logger(config, config_name)
    
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
    parser = argparse.ArgumentParser(description='Train BinaryErrorClassifier')
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
    
    # Extract config name from path
    config_path = Path(args.config)
    config_name = config_path.stem
    
    print(f"Training with configuration: {config_name}")
    print(f"Config file: {args.config}")
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    
    # Setup trainer
    print("Setting up trainer...")
    trainer = setup_trainer(config, config_name)
    
    # Print model summary
    print("\nModel Summary:")
    print(f"Architecture: {config['model']['arch_name']}")
    print(f"Dataset: {config['model']['dataset_name']}")
    print(f"Prediction Model: {config['model']['pred_model_name']}")
    print(f"Threshold: {config['model']['threshold_km']} km")
    print(f"Learning Rate: {config['model']['learning_rate']}")
    print(f"Batch Size: {config['model']['batch_size']}")
    print(f"\nTraining Configuration:")
    print(f"Max Steps: {config['trainer'].get('max_steps', 'N/A')}")
    print(f"Max Epochs: {config['trainer'].get('max_epochs', 'N/A')}")
    print(f"Validation Check Interval: {config['trainer'].get('val_check_interval', config['trainer'].get('check_val_every_n_epoch', 'N/A'))} {'steps' if 'val_check_interval' in config['trainer'] else 'epochs'}")
    print(f"Max Validation Batches: {config['trainer'].get('limit_val_batches', 'All')}")
    print(f"Max Test Batches: {config['trainer'].get('limit_test_batches', 'All')}")
    # Start training
    print("\nStarting training...")
    try:
        trainer.fit(
            model, 
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
        
        # Run test evaluation
        print("\nRunning test evaluation...")
        trainer.test(model, ckpt_path=best_model_path if best_model_path else None)
        print("Test evaluation completed!")
            
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()