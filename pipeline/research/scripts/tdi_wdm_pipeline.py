"""Reusable TDI/WDM post-processing pipeline.

Compatibility façade for the modular implementation in ``wdm_computation``.
"""

from __future__ import annotations

from .wdm_computation import (
    AET_LABELS,
    DEFAULT_WDM_RESOLUTIONS,
    GROUP_CHANNELS,
    LINK_LABELS,
    TDIWDMPipelineConfig,
    TDIWDMPipelinePaths,
    TDIWDMPipelineProducts,
    WDMResolution,
    XYZ_LABELS,
    run_tdi_wdm_pipeline,
)

__all__ = [
    "AET_LABELS",
    "DEFAULT_WDM_RESOLUTIONS",
    "GROUP_CHANNELS",
    "LINK_LABELS",
    "TDIWDMPipelineConfig",
    "TDIWDMPipelinePaths",
    "TDIWDMPipelineProducts",
    "WDMResolution",
    "XYZ_LABELS",
    "run_tdi_wdm_pipeline",
]
