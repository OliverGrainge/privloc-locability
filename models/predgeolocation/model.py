"""
PyTorch Lightning model for predicting log(1 + e) of geolocation error.

This model takes an image as input and predicts the log(1 + e) of the geodetic error
between true and predicted coordinates. The objective is to learn a scalar score
that correlates with the actual geolocation error.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LinearLR, SequentialLR
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging

from .archs import load_arch_and_transform
from data.datasets import load_prediction_dataset


class GeolocationErrorPredictionModel(pl.LightningModule):
    """
    PyTorch Lightning model for predicting log(1 + e) of geolocation error.
    
    This model learns to predict the log-transformed geodetic error between
    true and predicted coordinates, which can be used as a confidence score
    or for error estimation.
    """
    
    def __init__(
        self,
        arch_name: str,
        dataset_name: str, 
        pred_model_name: str, 
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        scheduler: str = 'cosine',
        warmup_steps: int = 1000,
        min_lr: float = 1e-6,
        loss_alpha: float = 1.0,
        loss_beta: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 8,
        **arch_kwargs
    ):
        """
        Initialize the error prediction model.
        
        Args:
            arch_name: Name of the architecture to load
            dataset_name: Name of the predgeolocationdataset to load
            pred_model_name: Name of the prediction model to load
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            scheduler: Learning rate scheduler ('cosine', 'plateau', or 'none')
            warmup_steps: Number of warmup steps for cosine scheduler
            min_lr: Minimum learning rate for cosine scheduler
            loss_alpha: Weight for MSE loss component
            loss_beta: Weight for Huber loss component
            **arch_kwargs: Additional arguments for architecture loading
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Load the backbone architecture
        self.model, self.transform = load_arch_and_transform(arch_name, **arch_kwargs)
        
        # Loss function components
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.SmoothL1Loss()
        
        # Metrics tracking
        self.train_metrics = {}
        self.val_metrics = {}
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input image tensor [batch_size, channels, height, width]
            
        Returns:
            Predicted log(1 + e) error scores [batch_size, 1]
        """
        # Extract features using backbone
        error_score = self.model(x)
        
        return error_score
    
    def compute_loss(self, pred_scores: torch.Tensor, true_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute the combined loss for error prediction.
        
        Args:
            pred_scores: Predicted log(1 + e) scores [batch_size, 1]
            true_scores: True log(1 + e) scores [batch_size, 1]
            
        Returns:
            Combined loss value
        """
        # MSE loss for overall accuracy
        mse_loss = self.mse_loss(pred_scores, true_scores)
        
        # Huber loss for robustness to outliers
        huber_loss = self.huber_loss(pred_scores, true_scores)
        
        # Combined loss
        total_loss = self.hparams.loss_alpha * mse_loss + self.hparams.loss_beta * huber_loss
        
        return total_loss
    
    def compute_metrics(self, pred_scores: torch.Tensor, true_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute evaluation metrics.
        
        Args:
            pred_scores: Predicted log(1 + e) scores
            true_scores: True log(1 + e) scores
            
        Returns:
            Dictionary of metrics
        """
        # Convert back to actual errors for metrics
        pred_errors = torch.exp(pred_scores.squeeze()) - 1
        true_errors = torch.exp(true_scores.squeeze()) - 1
        
        # Mean Absolute Error
        mae = F.l1_loss(pred_errors, true_errors)
        
        # Mean Squared Error
        mse = F.mse_loss(pred_errors, true_errors)
        
        # Root Mean Squared Error
        rmse = torch.sqrt(mse)
        
        # Mean Absolute Percentage Error
        mape = torch.mean(torch.abs((true_errors - pred_errors) / (true_errors + 1e-8))) * 100
        
        # Correlation coefficient
        pred_flat = pred_errors.flatten()
        true_flat = true_errors.flatten()
        correlation = torch.corrcoef(torch.stack([pred_flat, true_flat]))[0, 1]
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'correlation': correlation
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Batch containing 'image', 'true_lat', 'true_lon', 'pred_lat', 'pred_lon'
            batch_idx: Batch index
            
        Returns:
            Training loss
        """
        images = batch['image']
        true_lat = batch['true_lat']
        true_lon = batch['true_lon']
        pred_lat = batch['pred_lat']
        pred_lon = batch['pred_lon']
        
        # Calculate true geodetic errors
        true_errors = self._calculate_geodetic_errors(true_lat, true_lon, pred_lat, pred_lon)
        true_scores = torch.log(1 + true_errors).unsqueeze(1)
        
        # Forward pass
        pred_scores = self(images)
        
        # Compute loss
        loss = self.compute_loss(pred_scores, true_scores)
        
        # Compute metrics
        metrics = self.compute_metrics(pred_scores, true_scores)
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        for key, value in metrics.items():
            self.log(f'train/{key}', value, on_step=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step.
        
        Args:
            batch: Batch containing 'image', 'true_lat', 'true_lon', 'pred_lat', 'pred_lon'
            batch_idx: Batch index
            
        Returns:
            Validation loss
        """
        images = batch['image']
        true_lat = batch['true_lat']
        true_lon = batch['true_lon']
        pred_lat = batch['pred_lat']
        pred_lon = batch['pred_lon']
        
        # Calculate true geodetic errors
        true_errors = self._calculate_geodetic_errors(true_lat, true_lon, pred_lat, pred_lon)
        true_scores = torch.log(1 + true_errors).unsqueeze(1)
        
        # Forward pass
        pred_scores = self(images)
        
        # Compute loss
        loss = self.compute_loss(pred_scores, true_scores)
        
        # Compute metrics
        metrics = self.compute_metrics(pred_scores, true_scores)
        
        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for key, value in metrics.items():
            self.log(f'val/{key}', value, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def _calculate_geodetic_errors(self, true_lat: torch.Tensor, true_lon: torch.Tensor, 
                                 pred_lat: torch.Tensor, pred_lon: torch.Tensor) -> torch.Tensor:
        """
        Calculate geodetic errors using Haversine formula.
        
        Args:
            true_lat, true_lon: True coordinates
            pred_lat, pred_lon: Predicted coordinates
            
        Returns:
            Geodetic errors in kilometers
        """
        # Convert to radians
        true_lat_rad = torch.deg2rad(true_lat)
        true_lon_rad = torch.deg2rad(true_lon)
        pred_lat_rad = torch.deg2rad(pred_lat)
        pred_lon_rad = torch.deg2rad(pred_lon)
        
        # Haversine formula
        dlat = pred_lat_rad - true_lat_rad
        dlon = pred_lon_rad - true_lon_rad
        
        a = torch.sin(dlat/2)**2 + torch.cos(true_lat_rad) * torch.cos(pred_lat_rad) * torch.sin(dlon/2)**2
        c = 2 * torch.asin(torch.sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371.0
        return c * r
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        """
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        if self.hparams.scheduler == 'cosine':
            # Calculate total steps for cosine annealing
            # Assuming we need to estimate total steps from max_epochs
            # This is a rough estimate - in practice, you might want to pass total_steps explicitly
            steps_per_epoch = len(self.train_dataloader()) if hasattr(self, 'trainer') and self.trainer else 1000
            total_steps = self.trainer.max_epochs * steps_per_epoch if hasattr(self, 'trainer') and self.trainer else 100000
            
            # Create warmup scheduler
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.hparams.warmup_steps
            )
            
            # Create cosine annealing scheduler
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - self.hparams.warmup_steps,
                eta_min=self.hparams.min_lr
            )
            
            # Combine warmup and cosine annealing
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.hparams.warmup_steps]
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        elif self.hparams.scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=self.hparams.min_lr
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            return optimizer
    
    def predict_error_score(self, image: torch.Tensor) -> torch.Tensor:
        """
        Predict error score for a single image.
        
        Args:
            image: Input image tensor [1, channels, height, width]
            
        Returns:
            Predicted log(1 + e) error score
        """
        self.eval()
        with torch.no_grad():
            return self(image)
    
    def predict_actual_error(self, image: torch.Tensor) -> torch.Tensor:
        """
        Predict actual geodetic error in kilometers.
        
        Args:
            image: Input image tensor [1, channels, height, width]
            
        Returns:
            Predicted geodetic error in kilometers
        """
        log_score = self.predict_error_score(image)
        return torch.exp(log_score) - 1
    
    def train_dataloader(self) -> DataLoader:
        """
        Create training dataloader.
        
        Returns:
            DataLoader for training data
        """
        # Load the prediction dataset
        dataset = load_prediction_dataset(
            dataset_name=self.hparams.dataset_name,
            model_name=self.hparams.pred_model_name,
            transform=self.transform,
            max_samples=None  # Use all training samples
        )
        
        # Create DataLoader with memory-optimized settings
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,  # Default batch size, can be made configurable
            shuffle=False,
            num_workers=self.hparams.num_workers,  # Reduced workers to avoid memory issues
            pin_memory=False,  # Disable pin_memory to save memory
            drop_last=True,
            prefetch_factor=1  # Must be None when num_workers=0
        )
        
        return dataloader
    
    def test_dataloader(self) -> DataLoader:
        """
        Create test dataloader.
        
        Returns:
            DataLoader for test data
        """
        # Load the prediction dataset
        dataset = load_prediction_dataset(
            dataset_name=self.hparams.dataset_name,
            model_name=self.hparams.pred_model_name,
            transform=self.transform,
            max_samples=None  # Use all test samples
        )
        
        # Create DataLoader with memory-optimized settings
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,  # Default batch size, can be made configurable
            shuffle=False,  # No shuffling for test data
            num_workers=self.hparams.num_workers,  # Reduced workers to avoid memory issues
            pin_memory=False,  # Disable pin_memory to save memory
            drop_last=False,  # Don't drop last batch for test data
            prefetch_factor=1  # Must be None when num_workers=0
        )
        
        return dataloader

