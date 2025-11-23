"""
Simplified PyTorch Lightning model for binary classification of geolocation errors.

Classifies whether the geodetic error between true and predicted coordinates
is below a specified distance threshold.

Enhanced with class imbalance handling and detailed logging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, List
import logging
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import io
from PIL import Image

from .archs import load_arch_and_transform
from data.datasets import load_prediction_dataset

logger = logging.getLogger(__name__)


class BinaryErrorClassifier(pl.LightningModule):
    """
    Binary classifier for geolocation error prediction with class imbalance handling.
    
    Classifies whether the error is below a specified distance threshold.
    Supports multiple strategies to handle class imbalance:
    - Weighted BCE loss
    - Focal loss
    - Class weights in loss computation
    """
    
    def __init__(
        self,
        arch_name: str,
        dataset_name: str,
        pred_model_name: str,
        threshold_km: float = 2500.0,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        num_workers: int = 8,
        use_weighted_loss: bool = True,
        weight_positive: float = 1.0,
        use_focal_loss: bool = False,
        focal_loss_alpha: float = 0.25,
        focal_loss_gamma: float = 2.0,
    ):
        """
        Initialize the binary error classifier.
        
        Args:
            arch_name: Name of the backbone architecture to load
            dataset_name: Name of the dataset to load
            pred_model_name: Name of the prediction model
            threshold_km: Distance threshold in kilometers (default: 2500)
            learning_rate: Learning rate for Adam optimizer
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            use_weighted_loss: Whether to use weighted BCE loss based on class distribution
            weight_positive: Manual weight for positive class (ignored if use_weighted_loss=True)
            use_focal_loss: Whether to use focal loss instead of BCE
            focal_loss_alpha: Alpha parameter for focal loss (balance between classes)
            focal_loss_gamma: Gamma parameter for focal loss (focusing parameter)
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Load backbone architecture
        self.model, self.transform = load_arch_and_transform(arch_name)
        
        # Loss function configuration
        self.use_weighted_loss = use_weighted_loss
        self.use_focal_loss = use_focal_loss
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        
        # Placeholder for class weights (will be computed from data if use_weighted_loss=True)
        self.pos_weight = None
        
        # Base BCE loss (will use pos_weight if provided)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        # For tracking class distribution
        self.train_pos_count = 0
        self.train_neg_count = 0
        self.val_pos_count = 0
        self.val_neg_count = 0
        
        # For collecting test predictions for PR curve
        self.test_predictions: List[torch.Tensor] = []
        self.test_targets: List[torch.Tensor] = []
        self.test_random_predictions: List[torch.Tensor] = []

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input image tensor [batch_size, channels, height, width]
            
        Returns:
            Logits [batch_size]
        """
        return self.model(x)
    
    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss for addressing class imbalance.
        
        Focal loss reduces the loss for easy examples and focuses on hard examples.
        
        Args:
            logits: Predicted logits [batch_size]
            targets: Binary targets [batch_size]
            
        Returns:
            Focal loss value
        """
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal loss: -alpha * (1 - pt)^gamma * ce_loss
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.focal_loss_gamma
        
        # Apply alpha weighting for class balance
        alpha_t = self.focal_loss_alpha * targets + (1 - self.focal_loss_alpha) * (1 - targets)
        
        focal = alpha_t * focal_weight * bce
        return focal.mean()
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss with class imbalance handling.
        
        Args:
            logits: Predicted logits [batch_size]
            targets: Binary targets [batch_size]
            
        Returns:
            Loss value
        """
        if self.use_focal_loss:
            return self._focal_loss(logits, targets)
        
        # Standard BCE with optional class weighting
        if self.pos_weight is not None:
            # Manual weighting: apply higher weight to positive class
            bce_losses = self.bce_loss(logits, targets)
            weights = torch.where(
                targets == 1,
                torch.full_like(targets, self.pos_weight),
                torch.ones_like(targets)
            )
            return (bce_losses * weights).mean()
        
        return F.binary_cross_entropy_with_logits(logits, targets)
    
    def compute_metrics(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute evaluation metrics (accuracy, precision, recall, F1).
        
        Args:
            logits: Predicted logits [batch_size]
            targets: Binary targets [batch_size]
            
        Returns:
            Dictionary of metrics
        """
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        # Accuracy
        accuracy = (preds == targets).float().mean()
        
        # Confusion matrix elements
        tp = ((preds == 1) & (targets == 1)).float().sum()
        fp = ((preds == 1) & (targets == 0)).float().sum()
        tn = ((preds == 0) & (targets == 0)).float().sum()
        fn = ((preds == 0) & (targets == 1)).float().sum()
        
        # Random baseline: generate random predictions (50% positive, 50% negative)
        random_preds = (torch.rand(targets.shape, device=targets.device) > 0.5).float()
        
        # Random accuracy computed from random predictions
        random_accuracy = (random_preds == targets).float().mean()
        
        # Random confusion matrix elements
        random_tp = ((random_preds == 1) & (targets == 1)).float().sum()
        random_fp = ((random_preds == 1) & (targets == 0)).float().sum()
        random_fn = ((random_preds == 0) & (targets == 1)).float().sum()
        
        # Random precision, recall, F1
        random_precision = random_tp / (random_tp + random_fp + 1e-8)
        random_recall = random_tp / (random_tp + random_fn + 1e-8)
        random_f1 = 2 * (random_precision * random_recall) / (random_precision + random_recall + 1e-8)
        
        # Precision, recall, F1
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return {
            'accuracy': accuracy,
            'random_accuracy': random_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'random_precision': random_precision,
            'random_recall': random_recall,
            'random_f1': random_f1,
        }
    
    def _calculate_geodetic_error(self, lat1: torch.Tensor, lon1: torch.Tensor,
                                   lat2: torch.Tensor, lon2: torch.Tensor) -> torch.Tensor:
        """
        Calculate geodetic error using Haversine formula.
        
        Args:
            lat1, lon1: True coordinates
            lat2, lon2: Predicted coordinates
            
        Returns:
            Distance in kilometers
        """
        lat1_rad = torch.deg2rad(lat1)
        lon1_rad = torch.deg2rad(lon1)
        lat2_rad = torch.deg2rad(lat2)
        lon2_rad = torch.deg2rad(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.asin(torch.sqrt(a))
        
        return 6371.0 * c  # Earth's radius in km
    
    def _log_class_distribution(self, targets: torch.Tensor, stage: str) -> None:
        """
        Log class distribution for monitoring class imbalance.
        
        Args:
            targets: Binary targets
            stage: 'train' or 'val'
        """
        num_positive = (targets == 1).sum().item()
        num_negative = (targets == 0).sum().item()
        total = len(targets)
        
        pos_ratio = 100.0 * num_positive / total if total > 0 else 0
        neg_ratio = 100.0 * num_negative / total if total > 0 else 0
        
        logger.info(
            f"{stage.upper()} Class Distribution - "
            f"Positive: {num_positive} ({pos_ratio:.2f}%), "
            f"Negative: {num_negative} ({neg_ratio:.2f}%), "
            f"Total: {total}"
        )
        
        self.log(f'{stage}/class_pos_ratio', pos_ratio / 100.0)
        self.log(f'{stage}/class_neg_ratio', neg_ratio / 100.0)
        
        # Track cumulative counts
        if stage == 'train':
            self.train_pos_count += num_positive
            self.train_neg_count += num_negative
        else:
            self.val_pos_count += num_positive
            self.val_neg_count += num_negative
    
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Batch with keys: 'image', 'true_lat', 'true_lon', 'pred_lat', 'pred_lon'
            batch_idx: Batch index
            
        Returns:
            Training loss
        """
        images = batch['image']
        
        # Calculate geodetic error
        error_km = self._calculate_geodetic_error(
            batch['true_lat'], batch['true_lon'],
            batch['pred_lat'], batch['pred_lon']
        )

        # Create binary targets based on threshold
        targets = (error_km <= self.hparams.threshold_km).float()
        
        # Log class distribution and baselines (once per epoch)
        self._log_class_distribution(targets, 'train')
    
        # Compute class weights if needed (first batch)
        if self.pos_weight is None and self.use_weighted_loss:
            num_positive = (targets == 1).sum().item()
            num_negative = (targets == 0).sum().item()
            if num_positive > 0:
                self.pos_weight = num_negative / num_positive
                logger.info(f"Computed pos_weight: {self.pos_weight:.4f}")
        
        # Forward pass
        logits = self(images)
        
        # Compute loss and metrics
        loss = self.compute_loss(logits, targets)
        metrics = self.compute_metrics(logits, targets)
        
        # Log metrics
        self.log('train/loss', loss, prog_bar=True)
        for key, value in metrics.items():
            self.log(f'train/{key}', value)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Validation step.
        
        Args:
            batch: Batch with keys: 'image', 'true_lat', 'true_lon', 'pred_lat', 'pred_lon'
            batch_idx: Batch index
        """
        images = batch['image']
        
        # Calculate geodetic error
        error_km = self._calculate_geodetic_error(
            batch['true_lat'], batch['true_lon'],
            batch['pred_lat'], batch['pred_lon']
        )
        
        # Create binary targets
        targets = (error_km <= self.hparams.threshold_km).float()
        
        # Log class distribution and baselines (once per epoch)
        self._log_class_distribution(targets, 'val')
        
        # Forward pass
        logits = self(images)
        
        # Compute loss and metrics
        loss = self.compute_loss(logits, targets)
        metrics = self.compute_metrics(logits, targets)
        
        # Log metrics
        self.log('val/loss', loss, prog_bar=True)
        for key, value in metrics.items():
            self.log(f'val/{key}', value)
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Test step with collection of predictions for PR curve.
        
        Args:
            batch: Batch with keys: 'image', 'true_lat', 'true_lon', 'pred_lat', 'pred_lon'
            batch_idx: Batch index
        """
        images = batch['image']
        
        # Calculate geodetic error
        error_km = self._calculate_geodetic_error(
            batch['true_lat'], batch['true_lon'],
            batch['pred_lat'], batch['pred_lon']
        )
        
        # Create binary targets
        targets = (error_km <= self.hparams.threshold_km).float()
        
        # Log class distribution and baselines
        self._log_class_distribution(targets, 'test')
        
        # Forward pass
        logits = self(images)
        
        # Compute loss and metrics
        loss = self.compute_loss(logits, targets)
        metrics = self.compute_metrics(logits, targets)
        
        # Log metrics
        self.log('test/loss', loss, prog_bar=True)
        for key, value in metrics.items():
            self.log(f'test/{key}', value)
        
        # Store predictions and targets for PR curve
        probs = torch.sigmoid(logits)
        self.test_predictions.append(probs.detach().cpu())
        self.test_targets.append(targets.detach().cpu())
        
        # Store random predictions for baseline PR curve
        random_probs = torch.rand(targets.shape, device=targets.device)
        self.test_random_predictions.append(random_probs.detach().cpu())
    
    def on_test_epoch_end(self) -> None:
        """
        Compute and log PR curve and AUPRC at the end of test epoch.
        """
        if len(self.test_predictions) == 0:
            logger.warning("No test predictions collected for PR curve")
            return
        
        # Concatenate all predictions and targets
        all_probs = torch.cat(self.test_predictions, dim=0).float().numpy()
        all_targets = torch.cat(self.test_targets, dim=0).float().numpy()
        all_random_probs = torch.cat(self.test_random_predictions, dim=0).float().numpy()
        
        # Compute precision-recall curve for model predictions
        precision, recall, thresholds = precision_recall_curve(all_targets, all_probs)
        auprc = auc(recall, precision)
        
        # Compute precision-recall curve for random predictions
        random_precision, random_recall, random_thresholds = precision_recall_curve(all_targets, all_random_probs)
        random_auprc = auc(random_recall, random_precision)
        
        # Log AUPRCs
        self.log('test/auprc', auprc, prog_bar=True)
        self.log('test/random_auprc', random_auprc, prog_bar=True)
        logger.info(f"Test AUPRC: {auprc:.4f}, Random AUPRC: {random_auprc:.4f}")
        
        # Create combined PR curve plot (model + random baseline)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, linewidth=2, label=f'Model (AUPRC = {auprc:.4f})', color='blue')
        ax.plot(random_recall, random_precision, linewidth=2, label=f'Random (AUPRC = {random_auprc:.4f})', 
                color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        # Log PR curve figure to WandB
        plt.tight_layout()
        if self.logger is not None and hasattr(self.logger, 'experiment'):
            try:
                import wandb
                self.logger.experiment.log({
                    "test/pr_curve": wandb.Image(fig)
                })
                logger.info("PR curve logged to WandB")
            except Exception as e:
                logger.warning(f"Failed to log PR curve to WandB: {e}")
        else:
            logger.warning("WandB logger not available, skipping PR curve logging")
        
        plt.close(fig)
        
        # Create combined RP curve plot (model + random baseline, swapped axes)
        # Note: The area under the curve is the same as AUPRC (just with axes swapped for visualization)
        fig_rp, ax_rp = plt.subplots(figsize=(8, 6))
        ax_rp.plot(precision, recall, linewidth=2, label=f'Model (AUPRC = {auprc:.4f})', color='blue')
        ax_rp.plot(random_precision, random_recall, linewidth=2, label=f'Random (AUPRC = {random_auprc:.4f})', 
                   color='red', linestyle='--', alpha=0.7)
        ax_rp.set_xlabel('Precision', fontsize=12)
        ax_rp.set_ylabel('Recall', fontsize=12)
        ax_rp.set_title('Recall-Precision Curve', fontsize=14)
        ax_rp.legend(loc='best', fontsize=10)
        ax_rp.grid(True, alpha=0.3)
        ax_rp.set_xlim([0.0, 1.0])
        ax_rp.set_ylim([0.0, 1.05])
        
        # Log RP curve figure to WandB
        plt.tight_layout()
        if self.logger is not None and hasattr(self.logger, 'experiment'):
            try:
                import wandb
                self.logger.experiment.log({
                    "test/rp_curve": wandb.Image(fig_rp)
                })
                logger.info("RP curve logged to WandB")
            except Exception as e:
                logger.warning(f"Failed to log RP curve to WandB: {e}")
        else:
            logger.warning("WandB logger not available, skipping RP curve logging")
        
        plt.close(fig_rp)
        
        # Clear stored predictions and targets
        self.test_predictions.clear()
        self.test_targets.clear()
        self.test_random_predictions.clear()
    
    def configure_optimizers(self):
        """Configure optimizer with cosine annealing scheduler."""
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        dataset = load_prediction_dataset(
            dataset_name=self.hparams.dataset_name,
            model_name=self.hparams.pred_model_name,
            transform=self.transform,
            max_shards = 10,
            split = "train",
            ratio = 0.9,
        )
        
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
        )


    def val_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        dataset = load_prediction_dataset(
            dataset_name=self.hparams.dataset_name,
            model_name=self.hparams.pred_model_name,
            transform=self.transform,
            max_shards = 10,
            split = "val",
            ratio = 0.9,
        )
        
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
        )


    def test_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        dataset = load_prediction_dataset(
            dataset_name=self.hparams.dataset_name,
            model_name=self.hparams.pred_model_name,
            transform=self.transform,
            max_shards = 10,
            split = "val",
            ratio = 0.9,
        )
        
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
        )