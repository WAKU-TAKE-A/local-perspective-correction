"""Verification tests for the estimation core (specification sections 6 and 9).

Distances are compared pairwise, which is invariant to the unobservable
gauge freedoms (rotation and translation of the rectified plane).
"""

import numpy as np
import pytest

from lpc.core import (
    InsufficientDataError,
    _group_plane_direction,
    apply_homography,
    canonicalize_orientation,
    estimate_homography,
    fill_directions,
    jacobian,
    local_scale,
)
from lpc.simulate import make_scene


def _raw_items(scene, keep_dir_for=()):
    items = []
    for m in scene.measurements:
        keep = m.element_id in keep_dir_for
        items.append({
            "x": m.x, "y": m.y, "element_id": m.element_id,
            "group_id": m.group_id,
            "dir_x": m.dir_x if keep else None,
            "dir_y": m.dir_y if keep else None,
        })
    return items


def _assert_dir_matches(filled_item, m):
    dot = abs(filled_item["dir_x"] * m.dir_x + filled_item["dir_y"] * m.dir_y)
    assert dot > 0.9999


def test_fill_directions_from_element_lines():
    scene = make_scene(angle_deg=90.0)
    filled, counts = fill_directions(_raw_items(scene))
    assert counts["line"] == len(scene.measurements)
    for f, m in zip(filled, scene.measurements):
        _assert_dir_matches(f, m)


def test_fill_directions_vanishing_point_fallback():
    scene = make_scene(angle_deg=90.0)
    items = _raw_items(scene)
    # Keep only the first point of element A3, with no direction given.
    a3 = [i for i, m in enumerate(scene.measurements) if m.element_id == "A3"]
    items = [it for i, it in enumerate(items) if i not in a3[1:]]
    filled, counts = fill_directions(items)
    assert counts["vanishing_point"] == 1
    target = scene.measurements[a3[0]]
    f = [f for f in filled if f["element_id"] == "A3"][0]
    dot = abs(f["dir_x"] * target.dir_x + f["dir_y"] * target.dir_y)
    assert dot > 0.999


def test_fill_directions_impossible_for_lone_point():
    scene = make_scene(angle_deg=90.0)
    m = scene.measurements[0]
    items = [{"x": m.x, "y": m.y, "element_id": "A1", "group_id": "A",
              "dir_x": None, "dir_y": None}]
    with pytest.raises(InsufficientDataError):
        fill_directions(items)


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


def _display_dir(H, measurements, group_id) -> np.ndarray:
    """Group direction as it appears on screen (the plane's Y is drawn up,
    the image's Y is down)."""
    u = _group_plane_direction(H, measurements, group_id)
    return np.array([u[0], -u[1]])


def test_orientation_is_canonicalized():
    scene = make_scene(angle_deg=90.0)
    result = estimate_homography(scene.measurements, theta_deg=90.0)
    dets = [float(np.linalg.det(jacobian(result.H, m.q)))
            for m in scene.measurements]
    # Y-down image into a Y-up display: a non-mirrored result has det < 0.
    assert np.mean(dets) < 0
    d_a = _display_dir(result.H, scene.measurements, "A")
    assert d_a @ np.array([0.0, 1.0]) > 0.95  # first group vertical, downward
    d_b = _display_dir(result.H, scene.measurements, "B")
    # The second group is horizontal; its arrow sign is a data convention.
    assert abs(d_b @ np.array([1.0, 0.0])) > 0.85


def test_canonicalize_repairs_mirrored_solution():
    scene = make_scene(angle_deg=90.0)
    result = estimate_homography(scene.measurements, theta_deg=90.0)
    H_mirrored = np.diag([-1.0, 1.0, 1.0]) @ result.H
    # A reflected homography fits exactly the same (mirror ambiguity).
    for m in scene.measurements:
        assert local_scale(H_mirrored, m.q, m.t) == pytest.approx(
            m.resolution, rel=1e-9)
    H_fixed, corrected = canonicalize_orientation(H_mirrored, scene.measurements)
    assert corrected
    dets = [float(np.linalg.det(jacobian(H_fixed, m.q)))
            for m in scene.measurements]
    assert np.mean(dets) < 0
    for m in scene.measurements:
        assert local_scale(H_fixed, m.q, m.t) == pytest.approx(
            m.resolution, rel=1e-9)
    d_a = _display_dir(H_fixed, scene.measurements, "A")
    assert d_a @ np.array([0.0, 1.0]) > 0.95
