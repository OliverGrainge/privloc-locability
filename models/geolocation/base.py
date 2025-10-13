"""
Base class for geolocation models.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn


class GeoLocationModel(nn.Module, ABC):
    """
    Abstract base class for geolocation models that predict latitude and longitude
    from input tensors.
    
    All geolocation models should inherit from this class and implement the forward method.
    """
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, *)
            
        Returns:
            Tensor of shape (batch_size, 2), where columns are
            [latitude, longitude] predictions in degrees:
              - latitude: predicted latitude in degrees [-90, 90]
              - longitude: predicted longitude in degrees [-180, 180]
        """
        pass
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions without gradient computation.

        Args:
            x: Input tensor of shape (batch_size, *)

        Returns:
            Tensor of shape (batch_size, 2) containing [latitude, longitude] predictions
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    @staticmethod
    def haversine_distance(
        lat1: torch.Tensor, 
        lon1: torch.Tensor, 
        lat2: torch.Tensor, 
        lon2: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the Haversine distance between two sets of coordinates.
        
        Args:
            lat1: Latitude of first point(s) in degrees
            lon1: Longitude of first point(s) in degrees
            lat2: Latitude of second point(s) in degrees
            lon2: Longitude of second point(s) in degrees
            
        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lat1_rad = torch.deg2rad(lat1)
        lon1_rad = torch.deg2rad(lon1)
        lat2_rad = torch.deg2rad(lat2)
        lon2_rad = torch.deg2rad(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.asin(torch.sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371.0
        
        return c * r
    
    def compute_distance_error(
        self,
        pred_lat: torch.Tensor,
        pred_lon: torch.Tensor,
        true_lat: torch.Tensor,
        true_lon: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the distance error between predicted and true locations.
        
        Args:
            pred_lat: Predicted latitude
            pred_lon: Predicted longitude
            true_lat: True latitude
            true_lon: True longitude
            
        Returns:
            Distance error in kilometers
        """
        return self.haversine_distance(pred_lat, pred_lon, true_lat, true_lon)
    
    def evaluate(
        self,
        x: torch.Tensor,
        true_lat: torch.Tensor,
        true_lon: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate the model on a batch of data.
        
        Args:
            x: Input tensor
            true_lat: True latitude values
            true_lon: True longitude values
            
        Returns:
            Tensor containing the distance errors in km for each item in the batch.
        """
        pred_lat, pred_lon = self.predict(x)
        distances = self.compute_distance_error(pred_lat, pred_lon, true_lat, true_lon)
        return distances
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': next(self.parameters()).device if total_params > 0 else 'cpu'
        }

