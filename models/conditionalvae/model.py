import os
import pytorch_lightning as pl
from torch.optim import Adam
from torchmetrics.classification import BinaryAUROC, BinaryROC
from typing import Dict, Tuple

import torch.nn.functional as F
from diffusers import AutoencoderKL
import torch
import torch.nn as nn

from .arch import build_encoder


class ConditionalVAEWithPretrainedDecoder(nn.Module):
    """
    Conditional VAE using Stable Diffusion's pretrained VAE.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        encoder_out_dim: int,
        latent_dim: int = 256,
        img_size: int = 224,
        img_channels: int = 3,
        num_classes: int = 2,
        vae_model: str = "stabilityai/sd-vae-ft-mse",
        freeze_vae_decoder: bool = True,
    ):
        super().__init__()
        
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes
        
        # Load pretrained VAE from Stable Diffusion
        # Use local_files_only=True to avoid network requests when model is cached
        # Use cache_dir from environment variables if set
        cache_dir = os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE")
        self.pretrained_vae = AutoencoderKL.from_pretrained(
            vae_model, 
            local_files_only=True,
            cache_dir=cache_dir
        )
        
        # Freeze VAE decoder if requested
        if freeze_vae_decoder:
            for param in self.pretrained_vae.decoder.parameters():
                param.requires_grad = False
            for param in self.pretrained_vae.post_quant_conv.parameters():
                param.requires_grad = False
        
        # Encoder to latent distribution
        self.fc_mu = nn.Linear(encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_out_dim, latent_dim)
        
        # Conditional embedding
        self.class_embedding = nn.Embedding(num_classes, latent_dim)
        
        # Map our latent space to SD VAE's latent space
        # SD VAE expects [B, 4, H/8, W/8] for 224x224 → [B, 4, 28, 28]
        sd_latent_spatial = img_size // 8
        sd_latent_channels = 4
        
        self.latent_adapter = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),  # latent + class embedding
            nn.ReLU(),
            nn.Linear(512, sd_latent_channels * sd_latent_spatial * sd_latent_spatial),
        )
        
        self.sd_latent_spatial = sd_latent_spatial
        self.sd_latent_channels = sd_latent_channels
        
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))
    
    def encode(self, x: torch.Tensor):
        """Encode using your custom encoder."""
        h = self.encoder(x)
        if len(h.shape) == 4:
            h = h.flatten(start_dim=1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, y: torch.Tensor):
        """Decode using pretrained SD VAE."""
        # Get class embedding
        y_embed = self.class_embedding(y)
        
        # Concatenate z and class embedding
        z_cond = torch.cat([z, y_embed], dim=1)
        
        # Map to SD VAE latent space
        sd_latent = self.latent_adapter(z_cond)
        sd_latent = sd_latent.view(
            -1, 
            self.sd_latent_channels, 
            self.sd_latent_spatial, 
            self.sd_latent_spatial
        )
        
        # Decode using pretrained VAE decoder
        recon = self.pretrained_vae.decode(sd_latent).sample
        
        # Ensure output is in [0, 1] range
        recon = torch.clamp(recon, 0, 1)
        
        return recon
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar


class ConditionalVAEClassifier(pl.LightningModule):
    """Updated to use pretrained VAE."""
    
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
        vae_model: str = "stabilityai/sd-vae-ft-mse",  # NEW
        freeze_vae_decoder: bool = True,  # NEW
    ):
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
        
        # Use new conditional VAE with pretrained decoder
        self.cvae = ConditionalVAEWithPretrainedDecoder(
            encoder=encoder,
            encoder_out_dim=encoder_out_dim,
            latent_dim=latent_dim,
            img_size=img_size,
            img_channels=img_channels,
            num_classes=2,
            vae_model=vae_model,
            freeze_vae_decoder=freeze_vae_decoder,
        )
        
        self.beta = beta
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
        
        Returns per-sample ELBO, reconstruction loss, and KL loss.
        """
        # Forward pass
        recon, mu, logvar = self.cvae(x, y)
        
        # Compute reconstruction loss in pixel space
        recon_loss = F.mse_loss(recon, x, reduction='none')
        recon_loss = recon_loss.view(x.size(0), -1).mean(dim=1)  # [B]
        
        # KL divergence: per-sample mean over latent dims
        kl_element = 1 + logvar - mu.pow(2) - logvar.exp()
        kl_loss = -0.5 * kl_element.mean(dim=1)  # [B]
        
        # ELBO (negative loss is log-likelihood)
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
        
        # Log to WandB
        import matplotlib.pyplot as plt
        import wandb
        
        fig, ax = plt.subplots()
        ax.plot(fpr.cpu(), tpr.cpu())
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC (AUC = {val_auc:.3f})")
        
        if self.trainer is not None and self.trainer.is_global_zero:
            self.trainer.val_roc_figure = fig
        
        self.logger.experiment.log({
            "val/roc_curve": wandb.Image(fig)
        })
        plt.close(fig)
        
        self.val_auroc.reset()
        self.val_roc.reset()
    
    def on_test_epoch_end(self):
        """Log test metrics and create visualizations."""
        test_auc = self.test_auroc.compute()
        self.log("test/auroc", test_auc, prog_bar=True)
        
        fpr, tpr, thresholds = self.test_roc.compute()
        
        # Log to WandB
        import matplotlib.pyplot as plt
        import wandb
        
        # ROC curve
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr.cpu(), tpr.cpu())
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f"Test ROC (AUC = {test_auc:.3f})")
        
        if self.trainer is not None and self.trainer.is_global_zero:
            self.trainer.test_roc_figure = fig_roc
        
        self.logger.experiment.log({
            "test/roc_curve": wandb.Image(fig_roc)
        })
        
        # Don't close the figure here - let test.py collect and save it
        
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