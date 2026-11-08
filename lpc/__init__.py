"""Local perspective correction.

Estimate a planar homography from directional local-resolution
measurements (specification.md, v0.5).
"""

from .core import (
    EstimateResult,
    InsufficientDataError,
    Measurement,
    apply_homography,
    check_measurements,
    estimate_homography,
    jacobian,
    local_scale,
    warp_image,
)

__all__ = [
    "EstimateResult",
    "InsufficientDataError",
    "Measurement",
    "apply_homography",
    "check_measurements",
    "estimate_homography",
    "jacobian",
    "local_scale",
    "warp_image",
]

__version__ = "0.1.0"
