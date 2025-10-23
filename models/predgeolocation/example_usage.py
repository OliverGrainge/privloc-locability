"""
Example usage of the ErrorPredictionModel for predicting log(1 + e) of geolocation error.
"""

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from .model import ErrorPredictionModel
from .archs import load_arch


def create_model(arch_name: str = 'resnet', **kwargs) -> ErrorPredictionModel:
    """
    Create an error prediction model.
    
    Args:
        arch_name: Architecture name ('simple_cnn', 'resnet', 'vit')
        **kwargs: Additional model parameters
        
    Returns:
        Initialized ErrorPredictionModel
    """
    return ErrorPredictionModel(
        arch_name=arch_name,
        learning_rate=1e-4,
        weight_decay=1e-5,
        scheduler='cosine',
        **kwargs
    )


def train_model(
    model: ErrorPredictionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int = 100,
    save_dir: str = 'checkpoints'
) -> Trainer:
    """
    Train the error prediction model.
    
    Args:
        model: ErrorPredictionModel instance
        train_loader: Training data loader
        val_loader: Validation data loader
        max_epochs: Maximum number of epochs
        save_dir: Directory to save checkpoints
        
    Returns:
        Trained model
    """
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename='error-pred-{epoch:02d}-{val_loss:.2f}',
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val/loss',
        patience=10,
        mode='min',
        verbose=True
    )
    
    # Setup logger
    logger = TensorBoardLogger('logs', name='error_prediction')
    
    # Create trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        accelerator='auto',
        devices='auto',
        precision=16,  # Use mixed precision for efficiency
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    return trainer


def predict_errors(model: ErrorPredictionModel, images: torch.Tensor) -> torch.Tensor:
    """
    Predict geodetic errors for a batch of images.
    
    Args:
        model: Trained ErrorPredictionModel
        images: Batch of images [batch_size, channels, height, width]
        
    Returns:
        Predicted geodetic errors in kilometers
    """
    model.eval()
    with torch.no_grad():
        return model.predict_actual_error(images)


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_model(arch_name='resnet')
    
    # Example with dummy data
    batch_size = 8
    images = torch.randn(batch_size, 3, 224, 224)
    
    # Predict errors
    predicted_errors = predict_errors(model, images)
    print(f"Predicted errors: {predicted_errors}")
    
    # Example training setup (would need actual data loaders)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # trainer = train_model(model, train_loader, val_loader)
