"""
Prediction-based geolocation models for error estimation.
"""

from .model import GeolocationErrorPredictionModel

__all__ = [
    'GeolocationErrorPredictionModel',
    'load_arch',
    'DINOv2',
    'SigLip'

]
