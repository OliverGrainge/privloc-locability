#!/usr/bin/env python3
"""
Training script for GeolocationErrorPredictionModel.

This script loads a configuration file and trains the error prediction model
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
import torch

from models.predgeolocation.model import GeolocationErrorPredictionModel


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
        ModelCheckpoint callback
    """
    if not config.get('checkpoint', False):
        return None
    
    # Create checkpoint directory structure: checkpoints/{config_name}/
    checkpoint_dir = f"checkpoints/{config_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best_model-{epoch:02d}-{val_loss:.4f}',
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
        project="GeoErrorPredict",
        name=f"{config_name}_training",
        save_dir="logs/",
        log_model=True
    )
    
    return logger


def setup_early_stopping() -> EarlyStopping:
    """
    Setup early stopping callback.
    
    Returns:
        EarlyStopping callback
    """
    return EarlyStopping(
        monitor='val/loss',
        patience=10,
        mode='min',
        verbose=True
    )


def create_model(config: Dict[str, Any]) -> GeolocationErrorPredictionModel:
    """
    Create the error prediction model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized GeolocationErrorPredictionModel
    """
    model_config = config['model']
    
    # Pass all model configuration parameters
    model = GeolocationErrorPredictionModel(**model_config)
    
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
    
    # Add early stopping
    callbacks.append(setup_early_stopping())
    
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
    parser = argparse.ArgumentParser(description='Train GeolocationErrorPredictionModel')
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
    
    # Extract config name from path (e.g., 'train/test_train.yaml' -> 'test_train')
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
    print(f"Learning Rate: {config['model']['learning_rate']}")
    print(f"Scheduler: {config['model']['scheduler']}")
    print(f"Warmup Steps: {config['model']['warmup_steps']}")
    print(f"Max Steps: {config['trainer']['max_steps']}")
    
    # Start training
    print("\nStarting training...")
    try:
        trainer.fit(
            model, 
            ckpt_path=args.resume_from_checkpoint if args.resume_from_checkpoint else None
        )
        print("Training completed successfully!")
        
        # Print best model path if checkpointing was enabled
        if config.get('checkpoint', False) and hasattr(trainer.callbacks, 'ModelCheckpoint'):
            best_model_path = trainer.callbacks[0].best_model_path
            print(f"Best model saved at: {best_model_path}")
            
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
