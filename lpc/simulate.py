"""Synthetic data generation for verification (spec section 9).

A plane carrying two groups of parallel linear elements is projected by
a pinhole camera.  Each element carries a periodic pattern of known
pitch, so a measurement at a point on an element yields the directional
local resolution along that element.

Plane geometry: group A elements run along unit vector e_u, group B
elements along unit vector e_v, with angle phi between e_u and e_v.
A plane point is p = a*e_u + b*e_v (a, b in mm).  Elements of group A
sit at constant b offsets; elements of group B at constant a offsets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .core import Measurement


@dataclass
class SyntheticScene:
    measurements: list[Measurement]
    plane_points: dict[int, np.ndarray]  # measurement index -> true Euclidean plane coords (mm)
    image_size: tuple[int, int]
    true_angle_deg: float
    pitch_mm: float


def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_scene(
    angle_deg: float = 90.0,
    tilt_x_deg: float = -18.0,
    tilt_y_deg: float = 28.0,
    distance_mm: float = 1100.0,
    focal_px: float = 820.0,
    image_size: tuple[int, int] = (640, 480),
    pitch_mm: float = 10.0,
    a_offsets: tuple[float, ...] = (-260.0, 0.0, 260.0),
    b_offsets: tuple[float, ...] = (-190.0, 0.0, 190.0),
    samples_along: tuple[float, ...] = (-170.0, 0.0, 170.0),
    noise_rel: float = 0.0,
    drop_elements: frozenset[str] = frozenset(),
    seed: int = 0,
) -> SyntheticScene:
    """Build a synthetic scene.

    drop_elements: element ids to omit, e.g. {"A2"} drops the middle
    group-A element (missing-pattern tests, spec section 9.3).
    """
    rng = np.random.default_rng(seed)
    phi = math.radians(angle_deg)
    e_u = np.array([1.0, 0.0])
    e_v = np.array([math.cos(phi), math.sin(phi)])

    R = _rotation_matrix(math.radians(tilt_x_deg), math.radians(tilt_y_deg), 0.0)
    w_img, h_img = image_size
    K = np.array([[focal_px, 0, w_img / 2.0],
                  [0, focal_px, h_img / 2.0],
                  [0, 0, 1.0]])

    def project(p2: np.ndarray) -> np.ndarray:
        # Plane point (z = 0 in plane coords) -> camera coords -> pixels.
        p3 = np.array([p2[0], p2[1], 0.0])
        cam = R @ p3 + np.array([0.0, 0.0, distance_mm])
        img = K @ cam
        return img[:2] / img[2]

    measurements: list[Measurement] = []
    plane_points: dict[int, np.ndarray] = {}

    def add_point(p2: np.ndarray, direction_plane: np.ndarray,
                  element_id: str, group_id: str) -> None:
        q = project(p2)
        if not (0 <= q[0] < w_img and 0 <= q[1] < h_img):
            return
        # Directional scale along a unit plane direction:
        # pixels per mm = |d project / d s| where s is arc length.
        eps = 0.01
        dq = (project(p2 + eps * direction_plane)
              - project(p2 - eps * direction_plane)) / (2 * eps)
        px_per_mm = float(np.linalg.norm(dq))
        if px_per_mm <= 0:
            return
        r = 1.0 / px_per_mm
        if noise_rel > 0:
            r *= float(1.0 + rng.normal(0.0, noise_rel))
        t = dq / px_per_mm
        idx = len(measurements)
        measurements.append(Measurement(
            x=float(q[0]), y=float(q[1]), resolution=float(r),
            element_id=element_id, group_id=group_id,
            dir_x=float(t[0]), dir_y=float(t[1]),
        ))
        plane_points[idx] = p2.copy()

    # Group A: elements along e_u at constant b (offset along e_v).
    for i, b in enumerate(b_offsets):
        eid = f"A{i + 1}"
        if eid in drop_elements:
            continue
        for a in samples_along:
            add_point(a * e_u + b * e_v, e_u, eid, "A")
    # Group B: elements along e_v at constant a (offset along e_u).
    for j, a in enumerate(a_offsets):
        eid = f"B{j + 1}"
        if eid in drop_elements:
            continue
        for b in samples_along:
            add_point(a * e_u + b * e_v, e_v, eid, "B")

    return SyntheticScene(
        measurements=measurements,
        plane_points=plane_points,
        image_size=image_size,
        true_angle_deg=angle_deg,
        pitch_mm=pitch_mm,
    )
