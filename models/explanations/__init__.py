"""
Geoprivacy attribution and explanation methods.

This module provides tools for explaining which image regions contribute
most to geolocation vulnerability predictions.
"""

from .attribution import GeoAttributionHeatmap
from .visualization import (
    overlay_heatmap_on_image,
    create_attribution_figure,
    create_comparison_figure,
    save_attribution_visualization,
    get_top_regions_mask,
    highlight_top_regions
)

__all__ = [
    'GeoAttributionHeatmap',
    'overlay_heatmap_on_image',
    'create_attribution_figure',
    'create_comparison_figure',
    'save_attribution_visualization',
    'get_top_regions_mask',
    'highlight_top_regions'
]
