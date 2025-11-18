#!/usr/bin/env python3
"""
Training script for ConceptLocalizationPredictor.

This script loads a configuration file and trains the concept-based localization
predictor with checkpointing and logging capabilities.
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
from torch.utils.data import DataLoader

from models.conceptprediction import ConceptLocalizationPredictor
from data.datasets.predictiongeodataset import PredictionGeoDataset


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
        monitor='val_loss',
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
        project="ConceptLocalizationPredictor",
        name=f"{config_name}_training",
        save_dir="logs/",
        log_model=False
    )
    
    return logger





import math
import torch

def estimate_positive_ratio(train_dataset, target_distance_km: float) -> float:
    """
    Estimate the fraction of positives (localizable within target distance)
    on the *train* split, by scanning once.

    We replicate the Haversine logic from the model.
    """
    R = 6371.0  # Earth radius in km

    def haversine(lat1, lon1, lat2, lon2):
        lat1 = torch.deg2rad(lat1)
        lon1 = torch.deg2rad(lon1)
        lat2 = torch.deg2rad(lat2)
        lon2 = torch.deg2rad(lon2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.arcsin(torch.sqrt(a))
        return R * c

    n_pos = 0
    n_total = 0

    # train_dataset is a Subset, so index into its .dataset
    for idx in range(len(train_dataset)):
        sample = train_dataset[idx]
        true_lat = torch.tensor(sample["true_lat"], dtype=torch.float32)
        true_lon = torch.tensor(sample["true_lon"], dtype=torch.float32)
        pred_lat = torch.tensor(sample["pred_lat"], dtype=torch.float32)
        pred_lon = torch.tensor(sample["pred_lon"], dtype=torch.float32)

        dist = haversine(true_lat, true_lon, pred_lat, pred_lon)
        label = (dist <= target_distance_km).item()  # 1 or 0

        n_pos += int(label)
        n_total += 1

    if n_total == 0:
        raise ValueError("Train dataset is empty when estimating positive ratio.")
    return n_pos / n_total

    
def create_dataloaders(config: Dict[str, Any]) -> tuple:
    """
    Create train and validation dataloaders.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    data_config = config['data']
    dataset_path = data_config['dataset_path']
    batch_size = data_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 4)
    
    # Load dataset
    dataset = PredictionGeoDataset(dataset_path)
    
    # Check if embeddings are available
    if not dataset.has_embeddings:
        raise ValueError("Dataset does not contain pre-computed embeddings!")
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Split dataset into train/val
    train_size = int(data_config.get('train_split', 0.8) * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(data_config.get('seed', 42))
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    # >>> NEW: estimate positive ratio on train split
    target_distance_km = config["model"].get("target_distance_km", 25.0)
    pos_ratio = estimate_positive_ratio(train_dataset, target_distance_km)
    neg_ratio = 1.0 - pos_ratio
    pos_weight = neg_ratio / max(pos_ratio, 1e-6)  # avoid div by zero

    print(f"Estimated train positive ratio: {pos_ratio:.4f}")
    print(f"Using pos_weight = {pos_weight:.4f} for BCEWithLogitsLoss")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, pos_weight


def create_model(config: Dict[str, Any], pos_weight: float) -> ConceptLocalizationPredictor:
    model_config = config['model']
    concepts_config = config['concepts']

    model = ConceptLocalizationPredictor(
        text_concepts=concepts_config['text_concepts'],
        target_distance_km=model_config.get('target_distance_km', 25.0),
        num_prompts_per_concept=model_config.get('num_prompts_per_concept', 12),
        learning_rate=model_config.get('learning_rate', 1e-3),
        weight_decay=model_config.get('weight_decay', 1e-4),
        pos_weight=pos_weight,
        decision_threshold=model_config.get('decision_threshold', 0.5),
    )
    return model


def setup_trainer(config: Dict[str, Any], config_name: str, 
                 train_dataloader: DataLoader, val_dataloader: DataLoader) -> pl.Trainer:
    """
    Setup PyTorch Lightning trainer.
    
    Args:
        config: Configuration dictionary
        config_name: Name of the config file (without extension)
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader
        
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
    parser = argparse.ArgumentParser(description='Train ConceptLocalizationPredictor')
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
    
    # Extract config name from path (e.g., 'configs/concept_pred.yaml' -> 'concept_pred')
    config_path = Path(args.config)
    config_name = config_path.stem
    
    print(f"Training with configuration: {config_name}")
    print(f"Config file: {args.config}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_dataloader, val_dataloader, pos_weight = create_dataloaders(config)
    
    # Create model
    print("\nCreating model...")
    model = create_model(config, pos_weight)
    
    # Setup trainer
    print("\nSetting up trainer...")
    trainer = setup_trainer(config, config_name, train_dataloader, val_dataloader)
    
    # Print model summary
    print("\n" + "="*60)
    print("Model Configuration Summary")
    print("="*60)
    print(f"Text Concepts: {len(config['concepts']['text_concepts'])} concepts")
    print(f"  Sample concepts: {config['concepts']['text_concepts'][:3]}")
    print(f"Target Distance: {config['model']['target_distance_km']} km")
    print(f"Num Prompts per Concept: {config['model'].get('num_prompts_per_concept', 12)}")
    print(f"Learning Rate: {config['model']['learning_rate']}")
    print(f"Weight Decay: {config['model'].get('weight_decay', 1e-4)}")
    print(f"Class Weights: {config['model'].get('class_weights', None)}")
    print(f"Batch Size: {config['data']['batch_size']}")
    print(f"Max Epochs: {config['trainer'].get('max_epochs', 'N/A')}")
    print(f"Max Steps: {config['trainer'].get('max_steps', 'N/A')}")
    print("="*60)
    
    # Start training
    print("\nStarting training...")
    try:
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=args.resume_from_checkpoint if args.resume_from_checkpoint else None
        )
        print("\nTraining completed successfully!")
        
        # Print best model path if checkpointing was enabled
        if config.get('checkpoint', False) and checkpoint_callback:
            best_model_path = checkpoint_callback.best_model_path
            print(f"\nBest model saved at: {best_model_path}")
            
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise


if __name__ == "__main__":
    main()