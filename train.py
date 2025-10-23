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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
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
    
    model = GeolocationErrorPredictionModel(
        arch_name=model_config['arch_name'],
        dataset_name=model_config['dataset_name'],
        pred_model_name=model_config['pred_model_name'],
        learning_rate=model_config['learning_rate'],
        weight_decay=model_config['weight_decay'],
        scheduler=model_config['scheduler'],
        warmup_steps=model_config['warmup_steps'],
        min_lr=model_config['min_lr'],
        loss_alpha=model_config['loss_alpha'],
        loss_beta=model_config['loss_beta']
    )
    
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
    
    # Setup logger
    logger = setup_logger(config, config_name)
    
    # Create trainer
    trainer = pl.Trainer(
        max_steps=trainer_config['max_steps'],
        val_check_interval=trainer_config.get('val_check_interval', 1000),
        limit_val_batches=trainer_config.get('limit_val_batches', None),
        precision=trainer_config['precision'],
        accelerator=trainer_config['accelerator'],
        devices=trainer_config['devices'],
        strategy=trainer_config['strategy'],
        log_every_n_steps=trainer_config['log_every_n_steps'],
        check_val_every_n_epoch=trainer_config['check_val_every_n_epoch'],
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=trainer_config['enable_progress_bar'],
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
