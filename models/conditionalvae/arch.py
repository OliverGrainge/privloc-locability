"""
Utilities for constructing conditional VAE encoders.
"""

from typing import Tuple

import timm
import torch
import torch.nn as nn

__all__ = ["build_encoder"]


def build_encoder(
    encoder_name: str = "resnet18",
    img_size: int = 224,
    img_channels: int = 3,
    pretrained: bool = True,
) -> Tuple[nn.Module, int]:
    """
    Create a timm backbone and return it together with its flattened feature dim.

    Args:
        encoder_name: Name of timm model (e.g. 'resnet18', 'efficientnet_b0').
        img_size: Expected square input resolution.
        img_channels: Number of image channels.
        pretrained: Whether to load pretrained weights.

    Returns:
        (encoder, encoder_out_dim)
    """
    encoder = timm.create_model(
        encoder_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="",
    )

    with torch.no_grad():
        dummy_input = torch.randn(1, img_channels, img_size, img_size)
        encoder_out = encoder(dummy_input)

    if len(encoder_out.shape) == 4:
        # Use adaptive pooling so downstream code always receives flat features.
        encoder_out_dim = encoder_out.shape[1]
        encoder = nn.Sequential(
            encoder,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
    else:
        encoder_out_dim = encoder_out.shape[1]

    return encoder, encoder_out_dim


if __name__ == "__main__":
    enc, dim = build_encoder("resnet18")
    x = torch.randn(2, 3, 224, 224)
    feats = enc(x)
    print(f"encoder_out_dim={dim}, output_shape={feats.shape}")