"""
Sampling and Placement Strategy Modules
========================================

This package contains sampling strategies for object placement.
"""

from .heatmap_guided_placer import (
    HeatmapGuidedPlacer,
    HeatmapProcessor,
    CandidateExtractor,
    PoseFromLocationConverter,
)

__all__ = [
    "HeatmapGuidedPlacer",
    "HeatmapProcessor",
    "CandidateExtractor",
    "PoseFromLocationConverter",
]
