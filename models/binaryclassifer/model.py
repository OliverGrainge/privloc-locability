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
from typing import Dict, Tuple, Optional
from torchmetrics.classification import BinaryAUROC, BinaryROC
import logging



from .archs import load_arch_and_transform

logger = logging.getLogger(__name__)


class BinaryClassifier(pl.LightningModule):
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
        threshold_km: float = 2500.0,
        learning_rate: float = 1e-4,
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
            threshold_km: Distance threshold in kilometers (default: 2500)
            learning_rate: Learning rate for Adam optimizer
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
        
        # For tracking class distribution (for running average of class weights)
        self.train_pos_count = 0
        self.train_neg_count = 0
        self.num_batches_seen = 0  # Track number of batches for running average

        # For collecting test probabilities for distribution plot
        self.test_probs_list = []

        self._setup_metrics()

    def _setup_metrics(self): 
        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()
        self.val_roc = BinaryROC()   # will give you (fpr, tpr, thresholds)
        self.test_roc = BinaryROC()
        

        
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
        Compute evaluation metrics (accuracy, TPR, FPR).
        
        Args:
            logits: Predicted logits [batch_size]
            targets: Binary targets [batch_size]
            
        Returns:
            Dictionary with accuracy, true_positive_rate, and false_positive_rate
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
        
        # True Positive Rate (TPR) = Recall = TP / (TP + FN)
        tpr = tp / (tp + fn + 1e-8)
        
        # False Positive Rate (FPR) = FP / (FP + TN)
        fpr = fp / (fp + tn + 1e-8)
        
        return {
            'accuracy': accuracy,
            'TPR': tpr,
            'FPR': fpr,
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
    
        # Compute class weights using running average (more robust than first batch only)
        if self.use_weighted_loss and not self.use_focal_loss:
            num_positive = (targets == 1).sum().item()
            num_negative = (targets == 0).sum().item()
            
            # Update running counts
            self.train_pos_count += num_positive
            self.train_neg_count += num_negative
            self.num_batches_seen += 1
            
            # Compute weight from running average (after seeing at least 5 batches for stability)
            if self.num_batches_seen >= 5 and self.train_pos_count > 0:
                if self.pos_weight is None:
                    self.pos_weight = self.train_neg_count / self.train_pos_count
                    logger.info(f"Computed pos_weight from {self.num_batches_seen} batches: {self.pos_weight:.4f} "
                              f"(pos: {self.train_pos_count}, neg: {self.train_neg_count})")
                else:
                    # Update with running average
                    new_pos_weight = self.train_neg_count / self.train_pos_count
                    # Use exponential moving average for stability
                    self.pos_weight = 0.9 * self.pos_weight + 0.1 * new_pos_weight
        
        # Forward pass
        logits = self(images)
        
        # Compute loss and metrics
        loss = self.compute_loss(logits, targets)
        metrics = self.compute_metrics(logits, targets)

        
        # Log metrics
        self.log('train/loss', loss, prog_bar=True)
        for key, value in metrics.items():
            self.log(f'train/{key}', value)

        self.log("train/positive_rate", targets.mean(), on_epoch=True)
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
        
        # Forward pass
        logits = self(images)
        probs = torch.sigmoid(logits)
        
        # Compute loss and metrics
        loss = self.compute_loss(logits, targets)
        metrics = self.compute_metrics(logits, targets)

        self.val_auroc.update(probs, targets.int())
        self.val_roc.update(probs, targets.int())
        
        # Log metrics
        self.log('val/loss', loss, prog_bar=True)
        for key, value in metrics.items():
            self.log(f'val/{key}', value)
        self.log("val/positive_rate", targets.mean(), on_epoch=True)
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Test step.
        
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
        
        # Forward pass
        logits = self(images)
        probs = torch.sigmoid(logits)
        
        # Collect probabilities for distribution plot
        self.test_probs_list.append(probs.detach().cpu())
        
        # Compute loss and metrics
        loss = self.compute_loss(logits, targets)
        metrics = self.compute_metrics(logits, targets)

        self.test_auroc.update(probs, targets.int())
        self.test_roc.update(probs, targets.int())
        
        # Log metrics
        self.log('test/loss', loss, prog_bar=True)
        for key, value in metrics.items():
            self.log(f'test/{key}', value)
        self.log("test/positive_rate", targets.mean(), on_epoch=True)


    def on_validation_epoch_end(self):
        # AUROC scalar
        val_auc = self.val_auroc.compute()
        self.log("val/auroc", val_auc, prog_bar=True)

        # ROC curve points
        fpr, tpr, thresholds = self.val_roc.compute()

        # If using TensorBoard
        if isinstance(self.logger, pl.loggers.TensorBoardLogger):
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot(fpr.cpu(), tpr.cpu())
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC (AUC = {val_auc:.3f})")

            if self.trainer is not None and self.trainer.is_global_zero:
                self.trainer.val_roc_figure = fig  

            self.logger.experiment.add_figure(
                "val/roc_curve",
                fig,
                global_step=self.current_epoch,
            )
            plt.close(fig)

        # reset for next epoch
        self.val_auroc.reset()
        self.val_roc.reset()


    def on_test_epoch_end(self):
        test_auc = self.test_auroc.compute()
        self.log("test/auroc", test_auc, prog_bar=True)

        fpr, tpr, thresholds = self.test_roc.compute()

        # Create ROC curve figure regardless of logger type
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(fpr.cpu(), tpr.cpu())
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"Test ROC (AUC = {test_auc:.3f})")

        # Attach figure to trainer so test.py can collect and save it
        if self.trainer is not None and self.trainer.is_global_zero:
            self.trainer.test_roc_figure = fig  

        # Log to logger if it supports figures
        if self.logger is not None:
            if isinstance(self.logger, pl.loggers.TensorBoardLogger):
                self.logger.experiment.add_figure(
                    "test/roc_curve",
                    fig,
                    global_step=self.current_epoch,
                )
            elif hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log'):
                # For WandB and other loggers that support logging figures
                try:
                    import wandb
                    if isinstance(self.logger.experiment, wandb.sdk.wandb_run.Run):
                        self.logger.experiment.log({
                            "test/roc_curve": wandb.Image(fig)
                        })
                except (ImportError, AttributeError):
                    pass
        
        # Don't close the figure here - let test.py collect and save it

        # Create probability distribution histogram
        if len(self.test_probs_list) > 0:
            all_probs = torch.cat(self.test_probs_list, dim=0).to(torch.float32).numpy()
            
            fig_dist, ax_dist = plt.subplots(figsize=(8, 6))
            ax_dist.hist(all_probs, bins=50, edgecolor='black', alpha=0.7)
            ax_dist.set_xlabel("Predicted Probability")
            ax_dist.set_ylabel("Frequency")
            ax_dist.set_title("Distribution of Test Predictions")
            ax_dist.set_xlim(0, 1)
            ax_dist.grid(True, alpha=0.3)
            
            # Add statistics text
            mean_prob = all_probs.mean()
            std_prob = all_probs.std()
            min_prob = all_probs.min()
            max_prob = all_probs.max()
            stats_text = f"Mean: {mean_prob:.4f}\nStd: {std_prob:.4f}\nMin: {min_prob:.4f}\nMax: {max_prob:.4f}"
            ax_dist.text(0.7, 0.95, stats_text, transform=ax_dist.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            if self.trainer is not None and self.trainer.is_global_zero:
                self.trainer.test_prob_dist_figure = fig_dist

        self.test_auroc.reset()
        self.test_roc.reset()
        self.test_probs_list.clear()
    
    
    def configure_optimizers(self):
        """Configure optimizer with cosine annealing scheduler."""
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer