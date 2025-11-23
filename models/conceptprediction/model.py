"""
Concept-based Localization Predictor Trainer

This trainer computes concept similarities on-the-fly from image embeddings
and trains a linear probe to predict whether an image can be localized 
within a target distance.

The trainer expects:
- Image embeddings (from SigLip vision encoder)
- Text concepts (strings like "vegetation", "building", etc.)
- Ground truth and predicted coordinates
- A target distance threshold for binary classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import List

from data.datasets import load_prediction_dataset

logger = logging.getLogger(__name__)


def enhance_concept_prompts_ensemble(concepts, num_prompts=12):
    """
    Enhanced concept prompts using prompt ensembling to improve SigLip text-image matching.
    Uses multiple diverse templates to reduce variance and improve robustness.
    
    Args:
        concepts: List of basic concept strings
        num_prompts: Number of prompt templates to use per concept (default: 12)
    
    Returns:
        List of enhanced concept prompts (flattened list with multiple prompts per concept)
    """
    # Diverse prompt templates for better robustness
    prompt_templates = [
        "a photo of {}",
        "a satellite photo of {}",
        "an image of {}",
        "a landscape with {}",
        "a picture of {}",
        "a view of {}",
        "a scene with {}",
        "a photograph of {}",
        "aerial view of {}",
        "overhead view of {}",
        "ground level view of {}",
        "close-up of {}",
        "wide shot of {}",
        "detailed view of {}",
        "natural scene with {}",
        "urban scene with {}"
    ]
    
    # Select subset of templates if num_prompts is less than total templates
    if num_prompts < len(prompt_templates):
        # Use diverse selection from templates
        step = len(prompt_templates) // num_prompts
        selected_templates = prompt_templates[::step][:num_prompts]
    else:
        selected_templates = prompt_templates
    
    enhanced_concepts = []
    
    for concept in concepts:
        concept_lower = concept.lower()
        
        # Handle multi-word concepts (like "blue sky", "red car", etc.)
        if ' ' in concept_lower:
            # For multi-word concepts, use them as-is in templates
            for template in selected_templates:
                enhanced_concepts.append(template.format(concept))
        # Handle single-word concepts
        else:
            # Use appropriate article based on starting letter
            if concept_lower[0] in 'aeiou':
                concept_with_article = f"an {concept}"
            else:
                concept_with_article = f"a {concept}"
            
            for template in selected_templates:
                enhanced_concepts.append(template.format(concept_with_article))
    
    return enhanced_concepts


class ConceptLocalizationPredictor(pl.LightningModule):
    """
    Linear probe that predicts whether an image can be localized within 
    a target distance based on concept similarity scores computed on-the-fly.
    
    Args:
        text_concepts: List of text concept strings (e.g., ["vegetation", "building", "water"])
        target_distance_km: Distance threshold in kilometers for binary classification
        num_prompts_per_concept: Number of prompt templates per concept for ensembling
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        class_weights: Optional weights for handling class imbalance [weight_neg, weight_pos]
    """
    
    def __init__(
        self,
        text_concepts: list,
        dataset_name: str,
        pred_model_name: str,
        target_distance_km: float = 25.0,
        num_prompts_per_concept: int = 12,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        pos_weight: float = None,           # scalar ≈ n_neg / n_pos
        decision_threshold: float = 0.2,    # see below
        batch_size: int = 32,
        num_workers: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.text_concepts = text_concepts
        self.num_concepts = len(text_concepts)
        self.target_distance_km = float(target_distance_km)
        self.num_prompts_per_concept = num_prompts_per_concept
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.decision_threshold = float(decision_threshold)
        
        # Load SigLip model for text encoding
        from transformers import AutoModel, AutoProcessor
        model_name = "google/siglip-so400m-patch14-384"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        
        # Freeze text encoder (we only train the linear probe)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()
        
        # Pre-compute and cache text embeddings
        self.register_buffer('text_embeds_norm', self._compute_text_embeddings())
        
        # Linear probe: concept similarities -> binary classification
        self.linear_probe = nn.Linear(self.num_concepts, 1)
        
        # Loss function with optional class weights
        if pos_weight is not None:
            pw = torch.tensor(pos_weight, dtype=torch.float32)
            # register as buffer so it moves with .to(device)
            self.register_buffer("pos_weight", pw)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            self.pos_weight = None
            self.criterion = nn.BCEWithLogitsLoss()
        
        # Metrics for train/val/test
        self.train_accuracy = Accuracy(task='binary', threshold=self.decision_threshold)
        self.val_accuracy   = Accuracy(task='binary', threshold=self.decision_threshold)

        self.train_precision = Precision(task='binary', threshold=self.decision_threshold)
        self.val_precision   = Precision(task='binary', threshold=self.decision_threshold)

        self.train_recall = Recall(task='binary', threshold=self.decision_threshold)
        self.val_recall   = Recall(task='binary', threshold=self.decision_threshold)

        self.train_f1 = F1Score(task='binary', threshold=self.decision_threshold)
        self.val_f1   = F1Score(task='binary', threshold=self.decision_threshold)

        # AUROC remains threshold-free
        self.train_auroc = AUROC(task='binary')
        self.val_auroc   = AUROC(task='binary')
        
        # Separate metrics for random baseline
        self.train_random_accuracy = Accuracy(task='binary')
        self.val_random_accuracy = Accuracy(task='binary')
        
        self.train_random_precision = Precision(task='binary')
        self.val_random_precision = Precision(task='binary')
        
        self.train_random_recall = Recall(task='binary')
        self.val_random_recall = Recall(task='binary')
        
        self.train_random_f1 = F1Score(task='binary')
        self.val_random_f1 = F1Score(task='binary')
        
        self.train_random_auroc = AUROC(task='binary')
        self.val_random_auroc = AUROC(task='binary')
        
        # Test metrics
        self.test_accuracy = Accuracy(task='binary', threshold=self.decision_threshold)
        self.test_precision = Precision(task='binary', threshold=self.decision_threshold)
        self.test_recall = Recall(task='binary', threshold=self.decision_threshold)
        self.test_f1 = F1Score(task='binary', threshold=self.decision_threshold)
        self.test_auroc = AUROC(task='binary')
        
        self.test_random_accuracy = Accuracy(task='binary')
        self.test_random_precision = Precision(task='binary')
        self.test_random_recall = Recall(task='binary')
        self.test_random_f1 = F1Score(task='binary')
        self.test_random_auroc = AUROC(task='binary')
        
        # For collecting test predictions for PR curve
        self.test_predictions: List[torch.Tensor] = []
        self.test_targets: List[torch.Tensor] = []
        self.test_random_predictions: List[torch.Tensor] = []
    
    def _compute_text_embeddings(self):
        """
        Pre-compute text embeddings for all concepts using prompt ensembling.
        This is called once during initialization and cached as a buffer.
        
        Returns:
            text_embeds_norm: Normalized averaged text embeddings [num_concepts, embed_dim]
        """
        # Generate ensemble prompts for each concept
        ensemble_prompts = enhance_concept_prompts_ensemble(
            self.text_concepts, 
            num_prompts=self.num_prompts_per_concept
        )
        
        # Process all ensemble prompts
        text_inputs = self.processor(
            text=ensemble_prompts,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Get text embeddings for all prompts
        with torch.no_grad():
            text_embeds = self.text_encoder.get_text_features(**text_inputs)
            
            # Normalize embeddings (important for cosine similarity)
            text_embeds_norm = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            # Reshape to [num_concepts, num_prompts_per_concept, embed_dim]
            text_embeds_norm = text_embeds_norm.view(
                len(self.text_concepts), 
                self.num_prompts_per_concept, 
                -1
            )
            
            # Average embeddings across prompts for each concept
            text_embeds_avg = text_embeds_norm.mean(dim=1)
            
            # Re-normalize the averaged embeddings
            text_embeds_avg_norm = text_embeds_avg / text_embeds_avg.norm(dim=-1, keepdim=True)
        
        return text_embeds_avg_norm
    
    def compute_concept_similarities(self, image_embeds):
        """
        Compute similarity between image embeddings and pre-computed text embeddings.
        
        Args:
            image_embeds: Tensor of shape [batch_size, embed_dim] - SigLip vision embeddings
        
        Returns:
            similarities: Tensor of shape [batch_size, num_concepts]
        """
        # Normalize image embeddings
        image_embeds_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity via dot product
        similarities = image_embeds_norm @ self.text_embeds_norm.T
        
        return similarities
    
    def forward(self, image_embeds):
        """
        Forward pass: compute concept similarities then pass through linear probe.
        
        Args:
            image_embeds: Tensor of shape [batch_size, embed_dim] - SigLip vision embeddings
        
        Returns:
            logits: Tensor of shape [batch_size] with binary classification logits
        """
        # Compute concept similarities from image embeddings
        concept_similarities = self.compute_concept_similarities(image_embeds)
        
        # Pass through linear probe
        logits = self.linear_probe(concept_similarities).squeeze(-1)
        return logits
    
    def compute_localization_labels(self, true_lat, true_lon, pred_lat, pred_lon):
        """
        Compute binary labels based on localization error vs target distance.
        
        Args:
            true_lat, true_lon: Ground truth coordinates
            pred_lat, pred_lon: Predicted coordinates
        
        Returns:
            labels: Binary tensor (1 = localizable within target, 0 = not localizable)
        """
        # Haversine distance in kilometers
        def haversine_distance(lat1, lon1, lat2, lon2):
            lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
            c = 2 * torch.arcsin(torch.sqrt(a))
            return 6371.0 * c  # Earth radius in km
        
        errors = haversine_distance(true_lat, true_lon, pred_lat, pred_lon)
        labels = (errors <= self.target_distance_km).float()
        return labels, errors
    
    def shared_step(self, batch, batch_idx, stage='train'):
        """
        Shared step for train/val/test.
        
        Expected batch structure:
            - embedding: [batch_size, embed_dim] - SigLip vision embeddings
            - true_lat, true_lon: Ground truth coordinates
            - pred_lat, pred_lon: Predicted coordinates
        """
        image_embeds = batch['embedding']
        true_lat = batch['true_lat']
        true_lon = batch['true_lon']
        pred_lat = batch['pred_lat']
        pred_lon = batch['pred_lon']
        
        # Compute binary labels
        labels, errors = self.compute_localization_labels(
            true_lat, true_lon, pred_lat, pred_lon
        )
        
        # Forward pass (computes similarities internally)
        logits = self(image_embeds)
        
        # Compute loss
        loss = self.criterion(logits, labels)
        
        # Compute predictions
        probs = torch.sigmoid(logits)
        preds = (probs >= self.decision_threshold).float()
        random_probs = torch.rand(probs.shape).float().to(probs.device)
        random_preds = (random_probs >= 0.5).float()

        
        # Log metrics
        metrics_dict = {
            f'{stage}/loss': loss,
            f'{stage}/accuracy': getattr(self, f'{stage}_accuracy')(preds, labels),
            f'{stage}/precision': getattr(self, f'{stage}_precision')(preds, labels),
            f'{stage}/recall': getattr(self, f'{stage}_recall')(preds, labels),
            f'{stage}/f1': getattr(self, f'{stage}_f1')(preds, labels),
            f'{stage}/auroc': getattr(self, f'{stage}_auroc')(probs, labels),
            f'{stage}/prediction_ratio': preds.mean(),

            f'{stage}/random_accuracy': getattr(self, f'{stage}_random_accuracy')(random_preds, labels),
            f'{stage}/random_precision': getattr(self, f'{stage}_random_precision')(random_preds, labels),
            f'{stage}/random_recall': getattr(self, f'{stage}_random_recall')(random_preds, labels),
            f'{stage}/random_f1': getattr(self, f'{stage}_random_f1')(random_preds, labels),
            f'{stage}/random_auroc': getattr(self, f'{stage}_random_auroc')(random_probs, labels),

        }
        
        # Log class distribution for imbalance monitoring
        pos_ratio = labels.mean()
        metrics_dict[f'{stage}/positive_ratio'] = pos_ratio
        
        self.log_dict(metrics_dict, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage='train')
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage='val')
    
    def test_step(self, batch, batch_idx):
        """
        Test step with collection of predictions for PR curve.
        """
        image_embeds = batch['embedding']
        true_lat = batch['true_lat']
        true_lon = batch['true_lon']
        pred_lat = batch['pred_lat']
        pred_lon = batch['pred_lon']
        
        # Compute binary labels
        labels, errors = self.compute_localization_labels(
            true_lat, true_lon, pred_lat, pred_lon
        )
        
        # Forward pass (computes similarities internally)
        logits = self(image_embeds)
        
        # Compute loss
        loss = self.criterion(logits, labels)
        
        # Compute predictions
        probs = torch.sigmoid(logits)
        preds = (probs >= self.decision_threshold).float()
        random_probs = torch.rand(probs.shape).float().to(probs.device)
        random_preds = (random_probs >= 0.5).float()
        
        # Log metrics
        metrics_dict = {
            'test/loss': loss,
            'test/accuracy': self.test_accuracy(preds, labels),
            'test/precision': self.test_precision(preds, labels),
            'test/recall': self.test_recall(preds, labels),
            'test/f1': self.test_f1(preds, labels),
            'test/auroc': self.test_auroc(probs, labels),
            'test/prediction_ratio': preds.mean(),
            'test/random_accuracy': self.test_random_accuracy(random_preds, labels),
            'test/random_precision': self.test_random_precision(random_preds, labels),
            'test/random_recall': self.test_random_recall(random_preds, labels),
            'test/random_f1': self.test_random_f1(random_preds, labels),
            'test/random_auroc': self.test_random_auroc(random_probs, labels),
        }
        
        # Log class distribution for imbalance monitoring
        pos_ratio = labels.mean()
        metrics_dict['test/positive_ratio'] = pos_ratio
        
        self.log_dict(metrics_dict, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store predictions and targets for PR curve
        self.test_predictions.append(probs.detach().cpu())
        self.test_targets.append(labels.detach().cpu())
        self.test_random_predictions.append(random_probs.detach().cpu())
        
        return loss
    
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
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        return optimizer
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        dataset = load_prediction_dataset(
            dataset_name=self.hparams.dataset_name,
            model_name=self.hparams.pred_model_name,
            transform=None,  # No transform needed for embeddings
            max_shards=10,
            split="train",
            ratio=0.9,
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
        """Create validation dataloader."""
        dataset = load_prediction_dataset(
            dataset_name=self.hparams.dataset_name,
            model_name=self.hparams.pred_model_name,
            transform=None,  # No transform needed for embeddings
            max_shards=10,
            split="val",
            ratio=0.9,
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
        """Create test dataloader."""
        dataset = load_prediction_dataset(
            dataset_name=self.hparams.dataset_name,
            model_name=self.hparams.pred_model_name,
            transform=None,  # No transform needed for embeddings
            max_shards=10,
            split="val",
            ratio=0.9,
        )
        
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
        )