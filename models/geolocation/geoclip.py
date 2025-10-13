from geoclip import GeoCLIP
from models.geolocation.base import GeoLocationModel
import torch 
import torch.nn as nn
import torch.nn.functional as F

class GeoCLIPModel(GeoLocationModel):
    def __init__(self):
        super().__init__()
        geoclip = GeoCLIP()
        self.image_encoder = geoclip.image_encoder
        self.location_encoder = geoclip.location_encoder
        self.logit_scale = geoclip.logit_scale
        self.gps_gallery = geoclip.gps_gallery

    def forward(self, image):
        """ GeoCLIP's forward pass

        Args:
            image (torch.Tensor): Image tensor of shape (n, 3, 224, 224)

        Returns:
            lat (torch.Tensor): Latitude predictions of shape (n,)
            lon (torch.Tensor): Longitude predictions of shape (n,)
        """
        # Use gps_gallery as the location
        location = self.gps_gallery.to(image.device)

        # Compute Features
        image_features = self.image_encoder(image)
        location_features = self.location_encoder(location)
        logit_scale = self.logit_scale.exp()
        
        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        location_features = F.normalize(location_features, dim=1)
        
        # Cosine similarity (Image Features & Location Features)
        logits_per_image = logit_scale * (image_features @ location_features.t())

        probs_per_image = logits_per_image.softmax(dim=-1)
        top_pred = torch.topk(probs_per_image, 1, dim=1)
        
        # Get top GPS coordinates for all images in batch
        top_pred_gps = self.gps_gallery[top_pred.indices[:, 0].cpu()]
        
        lat = top_pred_gps[:, 0]
        lon = top_pred_gps[:, 1]
        
        return lat, lon

    def to(self, device):
        self.device = device
        self.image_encoder.to(device)
        self.location_encoder.to(device)
        self.logit_scale.data = self.logit_scale.data.to(device)
        return super().to(device)
