"""
Conditional VAE-based binary classifier for geolocation error prediction.

Uses a single conditional VAE that models p(x|y) where y is the binary 
localizability label. More efficient and elegant than dual VAE approach.

Key idea: Train one model conditioned on the class label, then use the 
conditional likelihoods p(x|y=1) and p(x|y=0) for classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from typing import Dict, Tuple
from torchmetrics.classification import BinaryAUROC, BinaryROC
import logging

from .arch import build_encoder

logger = logging.getLogger(__name__)


class ConditionalVAE(nn.Module):
    """
    Conditional VAE that models p(x|y) where y is binary class label.
    
    Architecture:
    - Encoder: x → z (shared across classes)
    - Decoder: [z, y] → x (conditioned on class label)
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        encoder_out_dim: int,
        latent_dim: int = 256,
        img_size: int = 224,
        img_channels: int = 3,
        num_classes: int = 2,
    ):
        """
        Initialize conditional VAE.
        
        Args:
            encoder: Pretrained encoder (from timm)
            encoder_out_dim: Output dimension of encoder
            latent_dim: Latent space dimension
            img_size: Image size
            img_channels: Number of image channels
            num_classes: Number of classes (2 for binary)
        """
        super().__init__()
        
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes
        
        # Encoder to latent distribution (shared)
        self.fc_mu = nn.Linear(encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_out_dim, latent_dim)
        
        # Conditional decoder
        # Input: [z, y_embedding]
        self.class_embedding = nn.Embedding(num_classes, latent_dim)
        
        # Decoder architecture
        self.init_size = img_size // 32  # 7 for 224
        self.init_channels = 512
        
        # Project [z + class_embedding] to initial feature map
        self.fc_decode = nn.Linear(latent_dim * 2, self.init_channels * self.init_size ** 2)
        
        # Transposed convolutions
        self.decoder = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(self.init_channels, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 28x28 -> 56x56
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 56x56 -> 112x112
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # 112x112 -> 224x224
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image to latent distribution.
        
        Args:
            x: Images [B, C, H, W]
            
        Returns:
            mu, logvar: Latent distribution parameters [B, latent_dim]
        """
        h = self.encoder(x)
        
        # Flatten if needed
        if len(h.shape) == 4:
            h = h.flatten(start_dim=1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector conditioned on class label.
        
        Args:
            z: Latent vectors [B, latent_dim]
            y: Class labels [B] (0 or 1)
            
        Returns:
            Reconstructed images [B, C, H, W]
        """
        # Get class embedding
        y_embed = self.class_embedding(y)  # [B, latent_dim]
        
        # Concatenate z and class embedding
        z_cond = torch.cat([z, y_embed], dim=1)  # [B, latent_dim * 2]
        
        # Project to feature map
        h = self.fc_decode(z_cond)
        h = h.view(-1, self.init_channels, self.init_size, self.init_size)
        
        # Decode
        return self.decoder(h)
    
    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            x: Images [B, C, H, W]
            y: Class labels [B]
            
        Returns:
            recon: Reconstructed images [B, C, H, W]
            mu: Latent means [B, latent_dim]
            logvar: Latent log variances [B, latent_dim]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar


class ConditionalVAEClassifier(pl.LightningModule):
    """
    Binary classifier using conditional VAE.
    
    Training: Learn p(x|y) via CVAE
    Inference: Classify using p(x|y=1) vs p(x|y=0) (via ELBO)
    """
    
    def __init__(
        self,
        encoder_name: str = "resnet18",
        latent_dim: int = 256,
        threshold_km: float = 2500.0,
        learning_rate: float = 1e-4,
        beta: float = 1.0,
        img_size: int = 224,
        img_channels: int = 3,
        pretrained: bool = True,
        freeze_encoder: bool = False,
    ):
        """
        Initialize conditional VAE classifier.
        
        Args:
            encoder_name: Name of timm backbone to load
            latent_dim: Latent space dimension
            threshold_km: Distance threshold for binary classification
            learning_rate: Learning rate
            beta: Beta parameter for VAE (KL weight)
            img_size: Input image size
            img_channels: Number of image channels
            pretrained: Whether to load pretrained timm weights
            freeze_encoder: Whether to freeze encoder weights
        """
        super().__init__()
        self.save_hyperparameters()

        encoder, encoder_out_dim = build_encoder(
            encoder_name=encoder_name,
            img_size=img_size,
            img_channels=img_channels,
            pretrained=pretrained,
        )

        if freeze_encoder:
            for param in encoder.parameters():
                param.requires_grad = False
        
        # Conditional VAE
        self.cvae = ConditionalVAE(
            encoder=encoder,
            encoder_out_dim=encoder_out_dim,
            latent_dim=latent_dim,
            img_size=img_size,
            img_channels=img_channels,
            num_classes=2,
        )
        
        self.beta = beta
        
        # Metrics
        self._setup_metrics()
        
        # For test visualization
        self.test_probs_list = []
        self.test_elbo_1_list = []
        self.test_elbo_0_list = []

        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))
    
    def _setup_metrics(self):
        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()
        self.val_roc = BinaryROC()
        self.test_roc = BinaryROC()
    
    def compute_elbo(
    self, 
    x: torch.Tensor, 
    y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute ELBO for given images and labels.

        Returns per-sample *mean* reconstruction and KL, so ELBO scale stays sane.
        """
        # Forward pass
        recon, mu, logvar = self.cvae(x, y)

        # Reconstruction loss: per-sample MEAN over all pixels/channels
        recon_loss = F.mse_loss(recon, x, reduction='none')
        recon_loss = recon_loss.view(x.size(0), -1).mean(dim=1)  # [B]

        # KL divergence: per-sample MEAN over latent dims
        kl_element = 1 + logvar - mu.pow(2) - logvar.exp()
        kl_loss = -0.5 * kl_element.mean(dim=1)  # [B]

        # ELBO (we’ll still treat this as "log-likelihood-ish")
        elbo = -(recon_loss + self.beta * kl_loss)  # [B]

        return elbo, recon_loss, kl_loss
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.
        
        Computes p(x|y=1) and p(x|y=0) via ELBO, then returns logits.
        
        Args:
            x: Images [B, C, H, W]
            
        Returns:
            logits: Classification logits [B]
        """
        batch_size = x.size(0)
        
        # Compute ELBO for y=1 (localizable)
        y_1 = torch.ones(batch_size, dtype=torch.long, device=x.device)
        elbo_1, _, _ = self.compute_elbo(x, y_1)
        
        # Compute ELBO for y=0 (non-localizable)
        y_0 = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        elbo_0, _, _ = self.compute_elbo(x, y_0)
        
        # Log-likelihood ratio as logits
        # log p(y=1|x) - log p(y=0|x) ≈ log p(x|y=1) - log p(x|y=0)
        #                               ≈ ELBO_1 - ELBO_0
        logits = elbo_1 - elbo_0
        logits = self.logit_scale * logits + self.logit_bias
        return logits
    
    def _calculate_geodetic_error(
        self, 
        lat1: torch.Tensor, lon1: torch.Tensor,
        lat2: torch.Tensor, lon2: torch.Tensor
    ) -> torch.Tensor:
        """Calculate geodetic error using Haversine formula."""
        lat1_rad = torch.deg2rad(lat1)
        lon1_rad = torch.deg2rad(lon1)
        lat2_rad = torch.deg2rad(lat2)
        lon2_rad = torch.deg2rad(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (torch.sin(dlat / 2) ** 2 + 
             torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2) ** 2)
        c = 2 * torch.asin(torch.sqrt(a))
        
        return 6371.0 * c
    
    def compute_metrics(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute evaluation metrics."""
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        accuracy = (preds == targets).float().mean()
        
        tp = ((preds == 1) & (targets == 1)).float().sum()
        fp = ((preds == 1) & (targets == 0)).float().sum()
        tn = ((preds == 0) & (targets == 0)).float().sum()
        fn = ((preds == 0) & (targets == 1)).float().sum()
        
        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        
        return {
            'accuracy': accuracy,
            'TPR': tpr,
            'FPR': fpr,
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step: Train conditional VAE on all images.
        
        Args:
            batch: Batch with keys 'image', 'true_lat', 'true_lon', 'pred_lat', 'pred_lon'
            batch_idx: Batch index
            
        Returns:
            Total loss
        """
        images = batch['image']
        
        # Calculate targets
        error_km = self._calculate_geodetic_error(
            batch['true_lat'], batch['true_lon'],
            batch['pred_lat'], batch['pred_lon']
        )
        targets = (error_km <= self.hparams.threshold_km).long()  # 0 or 1
        
        # Compute ELBO for actual labels (train CVAE)
        elbo, recon_loss, kl_loss = self.compute_elbo(images, targets)
        
        # VAE loss (we want to maximize ELBO, so minimize negative ELBO)
        vae_loss = recon_loss.mean() + self.beta * kl_loss.mean()
        
        # Classification loss (for monitoring and fine-tuning)
        logits = self.forward(images)
        clf_loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        
        # Total loss (primarily VAE loss)
        total_loss = vae_loss + 0.1 * clf_loss  # Small weight on classification
        
        # Metrics
        metrics = self.compute_metrics(logits, targets.float())
        
        # Logging
        self.log('train/loss', total_loss, prog_bar=True)
        self.log('train/vae_loss', vae_loss)
        self.log('train/recon_loss', recon_loss.mean())
        self.log('train/kl_loss', kl_loss.mean())
        self.log('train/clf_loss', clf_loss)
        for key, value in metrics.items():
            self.log(f'train/{key}', value)
        self.log('train/positive_rate', targets.float().mean(), on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        images = batch['image']
        
        error_km = self._calculate_geodetic_error(
            batch['true_lat'], batch['true_lon'],
            batch['pred_lat'], batch['pred_lon']
        )
        targets = (error_km <= self.hparams.threshold_km).float()
        
        # Get classification logits
        logits = self.forward(images)
        probs = torch.sigmoid(logits)
        
        # Losses
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        metrics = self.compute_metrics(logits, targets)
        
        # Update metrics
        self.val_auroc.update(probs, targets.int())
        self.val_roc.update(probs, targets.int())
        
        # Logging
        self.log('val/loss', loss, prog_bar=True)
        for key, value in metrics.items():
            self.log(f'val/{key}', value)
        self.log('val/positive_rate', targets.mean(), on_epoch=True)
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step."""
        images = batch['image']
        
        error_km = self._calculate_geodetic_error(
            batch['true_lat'], batch['true_lon'],
            batch['pred_lat'], batch['pred_lon']
        )
        targets = (error_km <= self.hparams.threshold_km).float()
        
        # Get classification logits and ELBOs
        batch_size = images.size(0)
        y_1 = torch.ones(batch_size, dtype=torch.long, device=images.device)
        y_0 = torch.zeros(batch_size, dtype=torch.long, device=images.device)
        
        elbo_1, _, _ = self.compute_elbo(images, y_1)
        elbo_0, _, _ = self.compute_elbo(images, y_0)
        logits = elbo_1 - elbo_0
        logits = self.logit_scale * logits + self.logit_bias
        probs = torch.sigmoid(logits)
        
        # Store for visualization
        self.test_probs_list.append(probs.detach().cpu())
        self.test_elbo_1_list.append(elbo_1.detach().cpu())
        self.test_elbo_0_list.append(elbo_0.detach().cpu())
        
        # Losses and metrics
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        metrics = self.compute_metrics(logits, targets)
        
        # Update metrics
        self.test_auroc.update(probs, targets.int())
        self.test_roc.update(probs, targets.int())
        
        # Logging
        self.log('test/loss', loss, prog_bar=True)
        for key, value in metrics.items():
            self.log(f'test/{key}', value)
        self.log('test/positive_rate', targets.mean(), on_epoch=True)
    
    def on_validation_epoch_end(self):
        """Log validation metrics at epoch end."""
        val_auc = self.val_auroc.compute()
        self.log("val/auroc", val_auc, prog_bar=True)
        
        fpr, tpr, thresholds = self.val_roc.compute()
        
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
        
        self.val_auroc.reset()
        self.val_roc.reset()
    
    def on_test_epoch_end(self):
        """Log test metrics and create visualizations."""
        test_auc = self.test_auroc.compute()
        self.log("test/auroc", test_auc, prog_bar=True)
        
        fpr, tpr, thresholds = self.test_roc.compute()
        
        import matplotlib.pyplot as plt
        
        # ROC curve
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr.cpu(), tpr.cpu())
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f"Test ROC (AUC = {test_auc:.3f})")
        
        if self.trainer is not None and self.trainer.is_global_zero:
            self.trainer.test_roc_figure = fig_roc
        
        if self.logger is not None and isinstance(self.logger, pl.loggers.TensorBoardLogger):
            self.logger.experiment.add_figure(
                "test/roc_curve",
                fig_roc,
                global_step=self.current_epoch,
            )
        
        # Probability distribution
        if len(self.test_probs_list) > 0:
            all_probs = torch.cat(self.test_probs_list, dim=0).numpy()
            
            fig_dist, ax_dist = plt.subplots(figsize=(8, 6))
            ax_dist.hist(all_probs, bins=50, edgecolor='black', alpha=0.7)
            ax_dist.set_xlabel("Predicted Probability")
            ax_dist.set_ylabel("Frequency")
            ax_dist.set_title("Distribution of Test Predictions")
            ax_dist.set_xlim(0, 1)
            ax_dist.grid(True, alpha=0.3)
            
            mean_prob = all_probs.mean()
            std_prob = all_probs.std()
            stats_text = f"Mean: {mean_prob:.4f}\nStd: {std_prob:.4f}"
            ax_dist.text(0.7, 0.95, stats_text, transform=ax_dist.transAxes,
                        verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            if self.trainer is not None and self.trainer.is_global_zero:
                self.trainer.test_prob_dist_figure = fig_dist
        
        # ELBO comparison plot
        if len(self.test_elbo_1_list) > 0:
            elbo_1 = torch.cat(self.test_elbo_1_list, dim=0).numpy()
            elbo_0 = torch.cat(self.test_elbo_0_list, dim=0).numpy()
            
            fig_elbo, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Scatter plot
            ax1.scatter(elbo_0, elbo_1, alpha=0.3, s=10)
            ax1.plot([elbo_0.min(), elbo_0.max()], 
                    [elbo_0.min(), elbo_0.max()], 
                    'r--', label='ELBO_1 = ELBO_0')
            ax1.set_xlabel("ELBO (y=0, non-localizable)")
            ax1.set_ylabel("ELBO (y=1, localizable)")
            ax1.set_title("ELBO Comparison")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Histogram of difference
            elbo_diff = elbo_1 - elbo_0
            ax2.hist(elbo_diff, bins=50, edgecolor='black', alpha=0.7)
            ax2.axvline(x=0, color='r', linestyle='--', label='Decision boundary')
            ax2.set_xlabel("ELBO_1 - ELBO_0")
            ax2.set_ylabel("Frequency")
            ax2.set_title("ELBO Difference Distribution")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if self.trainer is not None and self.trainer.is_global_zero:
                self.trainer.test_elbo_figure = fig_elbo
        
        self.test_auroc.reset()
        self.test_roc.reset()
        self.test_probs_list.clear()
        self.test_elbo_1_list.clear()
        self.test_elbo_0_list.clear()
    
    def configure_optimizers(self):
        """Configure optimizer."""
        return Adam(self.parameters(), lr=self.hparams.learning_rate)