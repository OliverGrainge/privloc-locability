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
from typing import Optional, Dict, Any, Tuple, List
import logging

from .archs import load_arch_and_transform
from data.datasets import load_prediction_dataset

# Configure logging
logger = logging.getLogger(__name__)


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
        head_type: str = 'regressor',
        K_km: Optional[List[float]] = None,
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
            head_type: Type of head ('regressor' or 'multik')
            K_km: List of distance thresholds in km for MultiKHead
            **arch_kwargs: Additional arguments for architecture loading
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Load the backbone architecture
        if head_type == 'multik':
            arch_kwargs['K_km'] = K_km
        self.model, self.transform = load_arch_and_transform(arch_name, head_type=head_type, **arch_kwargs)

        # Loss function components
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.SmoothL1Loss()
        if head_type == 'multik':
            self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Metrics tracking
        self.train_metrics = {}
        
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
    
    def compute_loss(self, pred_scores: torch.Tensor, true_scores: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the combined loss for error prediction, with additional stats.

        Args:
            pred_scores: Predicted scores [batch_size, 1] for regressor or [batch_size, m] for multik
            true_scores: True scores [batch_size, 1] for regressor or [batch_size, m] for multik
            
        Returns:
            Tuple of (loss_value, metrics_dict)
        """
        if self.hparams.head_type == 'multik':
            # MultiKHead: pred_scores and true_scores are [batch_size, m] logits
            # true_scores should be binary targets [batch_size, m]
            bce_loss = self.bce_loss(pred_scores, true_scores)
            
            return bce_loss, {'loss_bce': bce_loss}
        else:
            # Regressor head: pred_scores and true_scores are [batch_size, 1]
            pred_scores = pred_scores.squeeze()
            true_scores = true_scores.squeeze()
            
            # Element-wise MSE and Huber loss for statistics
            mse_loss_elem = (pred_scores - true_scores) ** 2
            huber_loss_elem = F.smooth_l1_loss(pred_scores, true_scores, reduction='none')

            # Reduce for the main loss values
            mse_loss = mse_loss_elem.mean()
            huber_loss = huber_loss_elem.mean()
            
            # Combined loss
            total_loss = self.hparams.loss_alpha * mse_loss + self.hparams.loss_beta * huber_loss

            # Prepare metrics for logging
            metrics = {
                'loss_mse': mse_loss,
                'loss_huber': huber_loss,
                'loss_alpha': self.hparams.loss_alpha,
                'loss_beta': self.hparams.loss_beta,
                'loss_mse_var': mse_loss_elem.var(unbiased=False),
                'loss_huber_var': huber_loss_elem.var(unbiased=False),
                'loss_mse_max': mse_loss_elem.max(),
                'loss_mse_min': mse_loss_elem.min(),
                'loss_huber_max': huber_loss_elem.max(),
                'loss_huber_min': huber_loss_elem.min(),
            }

            # Log prediction error statistics
            pred_error = torch.abs(pred_scores - true_scores)
            metrics.update({
                'pred_error_mean': pred_error.mean(),
                'pred_error_std': pred_error.std(),
                'pred_error_max': pred_error.max(),
                'pred_error_min': pred_error.min()
            })

            return total_loss, metrics
    
    def compute_metrics(self, pred_scores: torch.Tensor, true_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute evaluation metrics.
        
        Args:
            pred_scores: Predicted scores [batch_size, 1] for regressor or [batch_size, m] for multik
            true_scores: True scores [batch_size, 1] for regressor or [batch_size, m] for multik
            
        Returns:
            Dictionary of metrics
        """
        if self.hparams.head_type == 'multik':
            # MultiKHead: compute accuracy for each threshold
            pred_probs = torch.sigmoid(pred_scores)  # Convert logits to probabilities
            pred_binary = (pred_probs > 0.5).float()  # Convert to binary predictions
            
            # Compute accuracy for each threshold
            accuracies = []
            for k in range(pred_scores.shape[1]):
                acc = (pred_binary[:, k] == true_scores[:, k]).float().mean()
                accuracies.append(acc)
            
            # Overall accuracy (average across all thresholds)
            overall_acc = torch.stack(accuracies).mean()
            
            # Individual threshold accuracies
            metrics = {'overall_accuracy': overall_acc}
            for k, acc in enumerate(accuracies):
                metrics[f'acc_thresh_{k}'] = acc
            
            return metrics
        else:
            # Regressor head: convert back to actual errors for metrics
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
        # Extract batch data
        images = batch['image']
        true_lat = batch['true_lat']
        true_lon = batch['true_lon']
        pred_lat = batch['pred_lat']
        pred_lon = batch['pred_lon']
        
        # Log batch information
        self.log('train/batch_size', images.shape[0])
        self.log('train/image_mean', images.mean())
        self.log('train/image_std', images.std())
        
        # Calculate true geodetic errors
        true_errors = self._calculate_geodetic_errors(true_lat, true_lon, pred_lat, pred_lon)
        
        # Log error statistics
        self.log('train/true_errors_mean', true_errors.mean())
        self.log('train/true_errors_std', true_errors.std())
        self.log('train/true_errors_min', true_errors.min())
        self.log('train/true_errors_max', true_errors.max())
        
        if self.hparams.head_type == 'multik':
            # MultiKHead: create binary targets for each threshold
            K_km = self.model.head.K.to(true_errors.device)
            true_scores = (true_errors.unsqueeze(1) <= K_km.unsqueeze(0)).float()
            
            # Log MultiK specific statistics
            self.log('train/true_scores_mean', true_scores.mean())
            self.log('train/true_scores_std', true_scores.std())
            
            # Log threshold-specific statistics
            for k, threshold in enumerate(K_km):
                threshold_accuracy = true_scores[:, k].mean()
                self.log(f'train/threshold_{k}_accuracy', threshold_accuracy)
        else:
            # Regressor head: log-transformed errors
            true_scores = torch.log(1 + true_errors).unsqueeze(1)
            
            # Log regressor specific statistics
            self.log('train/true_scores_mean', true_scores.mean())
            self.log('train/true_scores_std', true_scores.std())
            self.log('train/true_scores_min', true_scores.min())
            self.log('train/true_scores_max', true_scores.max())
        
        # Forward pass
        pred_scores = self(images)
        
        # Log prediction statistics
        self.log('train/pred_scores_mean', pred_scores.mean())
        self.log('train/pred_scores_std', pred_scores.std())
        self.log('train/pred_scores_min', pred_scores.min())
        self.log('train/pred_scores_max', pred_scores.max())
        
        # Compute loss and get loss metrics
        loss, loss_metrics = self.compute_loss(pred_scores, true_scores)
        
        # Compute evaluation metrics
        eval_metrics = self.compute_metrics(pred_scores, true_scores)
        
        # Log main loss
        self.log('train/loss', loss, prog_bar=True)
        
        # Log all loss-related metrics
        for key, value in loss_metrics.items():
            self.log(f'train/{key}', value)
        
        # Log all evaluation metrics
        for key, value in eval_metrics.items():
            self.log(f'train/{key}', value)
        
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
                    'monitor': 'train/loss',
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
            Predicted log(1 + e) error score for regressor or logits for multik
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
            Predicted geodetic error in kilometers (only for regressor head)
        """
        if self.hparams.head_type == 'multik':
            raise ValueError("predict_actual_error not supported for MultiKHead. Use predict_recall_at_km instead.")
        
        log_score = self.predict_error_score(image)
        return torch.exp(log_score) - 1
    
    def predict_recall_at_km(self, image: torch.Tensor) -> torch.Tensor:
        """
        Predict recall probabilities at different distance thresholds.
        
        Args:
            image: Input image tensor [1, channels, height, width]
            
        Returns:
            Predicted probabilities [1, m] for each threshold
        """
        if self.hparams.head_type != 'multik':
            raise ValueError("predict_recall_at_km only supported for MultiKHead.")
        
        logits = self.predict_error_score(image)
        return torch.sigmoid(logits)
    
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
    

