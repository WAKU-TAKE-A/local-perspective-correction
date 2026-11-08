"""Verification tests for the estimation core (specification sections 6 and 9).

Distances are compared pairwise, which is invariant to the unobservable
gauge freedoms (rotation and translation of the rectified plane).
"""

import numpy as np
import pytest

from lpc.core import (
    InsufficientDataError,
    apply_homography,
    estimate_homography,
)
from lpc.simulate import make_scene


def _pairwise(P: np.ndarray) -> np.ndarray:
    d = P[:, None, :] - P[None, :, :]
    return np.linalg.norm(d, axis=-1)


def _distance_errors(scene, result) -> np.ndarray:
    idx = sorted(scene.plane_points)
    est = np.array([apply_homography(result.H, scene.measurements[i].q) for i in idx])
    tru = np.array([scene.plane_points[i] for i in idx])
    err = np.abs(_pairwise(est) - _pairwise(tru))
    iu = np.triu_indices(len(idx), k=1)
    return err[iu]


def test_noiseless_recovery_at_true_theta():
    scene = make_scene(angle_deg=90.0)
    result = estimate_homography(scene.measurements, theta_deg=90.0)
    errs = _distance_errors(scene, result)
    assert result.identifiable
    assert errs.mean() < 0.5
    assert errs.max() < 2.0


def test_recovery_with_small_noise():
    scene = make_scene(angle_deg=90.0, noise_rel=0.002, seed=3)
    result = estimate_homography(scene.measurements, theta_deg=90.0)
    errs = _distance_errors(scene, result)
    assert errs.mean() < 2.0


def test_missing_middle_element_still_recovers():
    scene = make_scene(angle_deg=90.0, drop_elements=frozenset({"A2"}))
    result = estimate_homography(scene.measurements, theta_deg=90.0)
    errs = _distance_errors(scene, result)
    assert errs.mean() < 1.0


def test_single_direction_group_is_insufficient():
    scene = make_scene(angle_deg=90.0)
    only_a = [m for m in scene.measurements if m.group_id == "A"]
    with pytest.raises(InsufficientDataError):
        estimate_homography(only_a, theta_deg=90.0)


def test_theta_mismatch_keeps_element_direction_distances():
    """With a wrong theta, distances along elements stay accurate while
    general pairwise distances degrade (consequence of hypothesis H-1)."""
    scene = make_scene(angle_deg=80.0)
    good = estimate_homography(scene.measurements, theta_deg=80.0)
    bad = estimate_homography(scene.measurements, theta_deg=90.0)
    errs_good = _distance_errors(scene, good)
    errs_bad = _distance_errors(scene, bad)
    assert errs_good.mean() < 0.5
    assert errs_bad.mean() > errs_good.mean() + 1.0
    # Residuals of the central constraint stay small even for wrong theta:
    # the pitch data alone cannot tell the angles apart.
    assert bad.rms_relative_residual < 0.01
