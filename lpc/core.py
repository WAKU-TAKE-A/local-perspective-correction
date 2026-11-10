"""Core estimation for local perspective correction.

Model (specification.md v0.5):
  - Each measurement gives a directional local resolution r_i (mm/pixel)
    at an image point q_i, measured along the direction t_i of the
    element the point belongs to.
  - The homography H (image -> plane) must satisfy the central constraint
        |J_H(q_i) t_i| = r_i
    where J_H is the Jacobian of the map q -> H(q).
  - Elements in the same direction group are parallel in the plane.
  - The angle theta between the two direction groups is NOT estimated;
    it is a user-supplied parameter.  estimate_homography() solves for
    H(theta) with theta fixed (spec section 6.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares


class InsufficientDataError(Exception):
    """Raised when the measurements cannot determine H(theta)."""


@dataclass
class Measurement:
    x: float
    y: float
    resolution: float  # mm / pixel, along (dir_x, dir_y)
    element_id: str
    group_id: str
    dir_x: float
    dir_y: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.x) and math.isfinite(self.y)):
            raise ValueError("measurement coordinates must be finite")
        if not math.isfinite(self.resolution) or self.resolution <= 0:
            raise ValueError("resolution must be a positive finite value")
        n = math.hypot(self.dir_x, self.dir_y)
        if not math.isfinite(n) or n == 0:
            raise ValueError("direction must be a non-zero finite vector")
        self.dir_x /= n
        self.dir_y /= n

    @property
    def q(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    @property
    def t(self) -> np.ndarray:
        return np.array([self.dir_x, self.dir_y], dtype=float)


# ---------------------------------------------------------------------------
# Homography basics
# ---------------------------------------------------------------------------

def apply_homography(H: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Map image point(s) q (..., 2) to plane point(s) (..., 2)."""
    q = np.asarray(q, dtype=float)
    hom = np.concatenate([q, np.ones(q.shape[:-1] + (1,))], axis=-1)
    out = hom @ H.T
    return out[..., :2] / out[..., 2:3]


def jacobian(H: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jacobian (2x2) of the map q -> H(q) at a single point q."""
    x, y = float(q[0]), float(q[1])
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    p = apply_homography(H, np.array([x, y]))
    A = H[:2, :2]
    c = H[2, :2]
    return (A - np.outer(p, c)) / w


def local_scale(H: np.ndarray, q: np.ndarray, t: np.ndarray) -> float:
    """|J_H(q) t| : mm per pixel along unit direction t at q."""
    return float(np.linalg.norm(jacobian(H, q) @ t))


# ---------------------------------------------------------------------------
# Input validation (spec sections 4.3 and 7.3)
# ---------------------------------------------------------------------------

def check_measurements(measurements: list[Measurement]) -> None:
    """Raise InsufficientDataError if estimation is clearly impossible."""
    if len(measurements) < 6:
        raise InsufficientDataError(
            f"測定点が少なすぎます（{len(measurements)} 点）。最低 6 点を目安にしてください。"
        )
    groups: dict[str, list[Measurement]] = {}
    for m in measurements:
        groups.setdefault(m.group_id, []).append(m)
    if len(groups) < 2:
        raise InsufficientDataError(
            "方向群が1つだけでは推定できません。測定方向が異なる2つの方向群が必要です。"
        )
    for gid, ms in groups.items():
        if len(ms) < 2:
            raise InsufficientDataError(
                f"方向群 {gid} の測定点が {len(ms)} 点しかありません。各方向群に最低2点が必要です。"
            )
    # Mean direction of each group must be well separated.
    means = {}
    for gid, ms in groups.items():
        v = np.mean([m.t for m in ms], axis=0)
        n = np.linalg.norm(v)
        means[gid] = v / n if n > 0 else v
    gids = list(means)
    for i in range(len(gids)):
        for j in range(i + 1, len(gids)):
            cosang = abs(float(means[gids[i]] @ means[gids[j]]))
            if cosang > math.cos(math.radians(20.0)):
                raise InsufficientDataError(
                    f"方向群 {gids[i]} と {gids[j]} の平均方向が近すぎます"
                    "（20度未満）。十分に離れた2方向が必要です。"
                )
    # Points must not be (nearly) collinear / clustered.
    pts = np.array([m.q for m in measurements])
    centered = pts - pts.mean(axis=0)
    sv = np.linalg.svd(centered, compute_uv=False)
    if sv[0] <= 0 or sv[1] < 1e-3 * sv[0]:
        raise InsufficientDataError(
            "測定点がほぼ一直線上または狭い範囲に集中しています。面内に広く分布させてください。"
        )


# ---------------------------------------------------------------------------
# Direction derivation
# ---------------------------------------------------------------------------

def _line_direction(points: np.ndarray) -> np.ndarray | None:
    """Direction of the straight line through points (PCA first axis)."""
    if len(points) < 2:
        return None
    centered = points - points.mean(axis=0)
    if np.linalg.norm(centered) < 1e-9:
        return None
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    d = vt[0]
    n = np.linalg.norm(d)
    return d / n if n > 0 else None


def _vanishing_point(lines: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray | None:
    """Least-squares intersection of lines given as (point, direction)."""
    if len(lines) < 2:
        return None
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for p, d in lines:
        P = np.eye(2) - np.outer(d, d)
        A += P
        b += P @ p
    try:
        if abs(np.linalg.det(A)) < 1e-12:
            return None
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


def fill_directions(items: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Fill missing measurement directions automatically.

    Each item is a dict with keys x, y, element_id, group_id and optional
    dir_x / dir_y (None or missing means "not given").  Derivation order:

    1. An element with two or more points lies on a straight image line,
       whose direction is the measurement direction for all its points.
    2. A single-point element without a given direction uses the
       direction toward the vanishing point of its direction group,
       computed from the lines of the other elements in the group.
    3. A given direction is used as-is when neither applies.

    Returns (filled items, counts per method).  Raises
    InsufficientDataError when a direction cannot be determined.
    """
    by_element: dict[str, list[int]] = {}
    for i, it in enumerate(items):
        by_element.setdefault(str(it["element_id"]), []).append(i)

    line_dir: dict[str, np.ndarray] = {}
    for eid, idxs in by_element.items():
        pts = np.array([[items[i]["x"], items[i]["y"]] for i in idxs], dtype=float)
        d = _line_direction(pts)
        if d is not None:
            line_dir[eid] = d

    # Vanishing point per group, from element lines of that group.
    group_lines: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for eid, d in line_dir.items():
        idxs = by_element[eid]
        rep = items[idxs[0]]
        group_lines.setdefault(str(rep["group_id"]), []).append(
            (np.array([rep["x"], rep["y"]], dtype=float), d))
    group_vp = {gid: _vanishing_point(lines) for gid, lines in group_lines.items()}

    counts = {"line": 0, "vanishing_point": 0, "given": 0}
    filled: list[dict] = []
    for i, it in enumerate(items):
        out = dict(it)
        eid = str(it["element_id"])
        gid = str(it["group_id"])
        given = it.get("dir_x") is not None and it.get("dir_y") is not None
        if eid in line_dir:
            d = line_dir[eid]
            out["dir_x"], out["dir_y"] = float(d[0]), float(d[1])
            counts["line"] += 1
        elif given:
            out["dir_x"], out["dir_y"] = float(it["dir_x"]), float(it["dir_y"])
            counts["given"] += 1
        else:
            vp = group_vp.get(gid)
            q = np.array([it["x"], it["y"]], dtype=float)
            if vp is not None and np.linalg.norm(vp - q) > 1e-6:
                d = (vp - q) / np.linalg.norm(vp - q)
                out["dir_x"], out["dir_y"] = float(d[0]), float(d[1])
                counts["vanishing_point"] += 1
            else:
                raise InsufficientDataError(
                    f"要素 {eid} の測定方向を決められません。その要素に2点以上を入れるか、"
                    "dir_x / dir_y を指定してください。"
                )
        filled.append(out)
    return filled, counts


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------

def _element_plane_direction(H: np.ndarray, ms: list[Measurement]) -> np.ndarray | None:
    """Direction of one element in the plane, from its transformed points."""
    if len(ms) < 2:
        return None
    ref = ms[0].t
    ordered = sorted(ms, key=lambda m: float(m.q @ ref))
    p0 = apply_homography(H, ordered[0].q)
    p1 = apply_homography(H, ordered[-1].q)
    d = p1 - p0
    n = np.linalg.norm(d)
    return d / n if n > 0 else None


def _group_plane_direction(H: np.ndarray, measurements: list[Measurement],
                           group_id: str) -> np.ndarray | None:
    """Mean plane direction of a direction group (sign-aligned)."""
    by_element: dict[str, list[Measurement]] = {}
    for m in measurements:
        if m.group_id == group_id:
            by_element.setdefault(m.element_id, []).append(m)
    dirs = []
    for ms in by_element.values():
        d = _element_plane_direction(H, ms)
        if d is not None:
            dirs.append(d)
    if not dirs:
        return None
    ref = dirs[0]
    aligned = [d if float(d @ ref) >= 0 else -d for d in dirs]
    mean = np.mean(aligned, axis=0)
    n = np.linalg.norm(mean)
    return mean / n if n > 0 else None


def _residuals(params: np.ndarray, measurements: list[Measurement],
               theta_deg: float, weight_parallel: float,
               weight_theta: float) -> np.ndarray:
    H = np.array([
        [params[0], params[1], params[2]],
        [params[3], params[4], params[5]],
        [params[6], params[7], 1.0],
    ])
    res: list[float] = []
    # Central constraint, relative residuals.
    for m in measurements:
        res.append((local_scale(H, m.q, m.t) - m.resolution) / m.resolution)
    # Parallelism within each group.
    group_ids = sorted({m.group_id for m in measurements})
    for gid in group_ids:
        by_element: dict[str, list[Measurement]] = {}
        for m in measurements:
            if m.group_id == gid:
                by_element.setdefault(m.element_id, []).append(m)
        dirs = []
        for ms in by_element.values():
            d = _element_plane_direction(H, ms)
            if d is not None:
                dirs.append(d)
        for d in dirs[1:]:
            res.append(weight_parallel * (1.0 - abs(float(d @ dirs[0]))))
    # Angle parameter theta between the two main groups.
    if len(group_ids) >= 2:
        ua = _group_plane_direction(H, measurements, group_ids[0])
        ub = _group_plane_direction(H, measurements, group_ids[1])
        if ua is not None and ub is not None:
            cos_target = math.cos(math.radians(theta_deg))
            res.append(weight_theta * (float(ua @ ub) - cos_target))
    return np.array(res, dtype=float)


def _numeric_jacobian(params: np.ndarray, measurements: list[Measurement],
                      theta_deg: float, weight_parallel: float,
                      weight_theta: float) -> np.ndarray:
    base = _residuals(params, measurements, theta_deg, weight_parallel, weight_theta)
    J = np.zeros((len(base), len(params)))
    for k in range(len(params)):
        step = 1e-6 * max(1.0, abs(params[k]))
        p2 = params.copy()
        p2[k] += step
        J[:, k] = (_residuals(p2, measurements, theta_deg,
                              weight_parallel, weight_theta) - base) / step
    return J


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

@dataclass
class EstimateResult:
    H: np.ndarray
    theta_deg: float
    converged: bool
    message: str
    rms_relative_residual: float
    max_relative_residual: float
    per_group_rms: dict[str, float] = field(default_factory=dict)
    jacobian_rank: int = 0
    jacobian_condition: float = float("inf")
    identifiable: bool = False


def estimate_homography(measurements: list[Measurement],
                        theta_deg: float = 90.0,
                        weight_parallel: float = 2.0,
                        weight_theta: float = 5.0) -> EstimateResult:
    """Estimate H(theta) from directional local-resolution measurements.

    Raises InsufficientDataError when the input clearly cannot determine
    the homography (spec section 7.3).
    """
    check_measurements(measurements)
    r_mean = float(np.mean([m.resolution for m in measurements]))

    # Initial guesses: isotropic scale plus a few projective tilts.
    inits = []
    for cx, cy in [(0.0, 0.0), (8e-4, 0.0), (-8e-4, 0.0), (0.0, 8e-4), (0.0, -8e-4)]:
        inits.append(np.array([r_mean, 0.0, 0.0,
                               0.0, r_mean, 0.0,
                               cx, cy]))
    best = None
    for p0 in inits:
        try:
            sol = least_squares(
                _residuals, p0,
                args=(measurements, theta_deg, weight_parallel, weight_theta),
                method="lm", max_nfev=4000,
            )
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        raise InsufficientDataError("最適化がすべて失敗しました。入力を確認してください。")

    params = best.x
    H = np.array([
        [params[0], params[1], params[2]],
        [params[3], params[4], params[5]],
        [params[6], params[7], 1.0],
    ])
    rel = np.array([(local_scale(H, m.q, m.t) - m.resolution) / m.resolution
                    for m in measurements])
    per_group: dict[str, float] = {}
    for gid in sorted({m.group_id for m in measurements}):
        vals = [r for r, m in zip(rel, measurements) if m.group_id == gid]
        per_group[gid] = float(np.sqrt(np.mean(np.square(vals)))) if vals else 0.0

    J = _numeric_jacobian(params, measurements, theta_deg,
                          weight_parallel, weight_theta)
    sv = np.linalg.svd(J, compute_uv=False)
    tol = (sv[0] * 1e-6) if len(sv) and sv[0] > 0 else 0.0
    rank = int(np.sum(sv > tol))
    nonzero = sv[sv > tol]
    condition = float(nonzero[0] / nonzero[-1]) if len(nonzero) else float("inf")
    # 8 parameters, 3 gauge freedoms (rotation + translation of the plane)
    # are unobservable, so a fully determined problem has rank 5.
    identifiable = rank >= 5

    return EstimateResult(
        H=H,
        theta_deg=theta_deg,
        converged=bool(best.success),
        message=str(best.message),
        rms_relative_residual=float(np.sqrt(np.mean(rel ** 2))),
        max_relative_residual=float(np.max(np.abs(rel))),
        per_group_rms=per_group,
        jacobian_rank=rank,
        jacobian_condition=condition,
        identifiable=identifiable,
    )


# ---------------------------------------------------------------------------
# Image warping
# ---------------------------------------------------------------------------

def warp_image(image, H: np.ndarray, px_per_mm: float = 2.0):
    """Warp a PIL image into the rectified plane coordinate system.

    H maps image pixels to plane millimetres.  The output uses
    px_per_mm pixels per millimetre, with the plane Y axis pointing up.
    Returns a new PIL.Image.
    """
    from PIL import Image
    from scipy.ndimage import map_coordinates

    img = np.asarray(image.convert("RGB"), dtype=float)
    h_img, w_img = img.shape[:2]
    corners = np.array([[0, 0], [w_img, 0], [0, h_img], [w_img, h_img]], dtype=float)
    plane = apply_homography(H, corners)
    lo = plane.min(axis=0)
    hi = plane.max(axis=0)
    out_w = max(2, int(math.ceil((hi[0] - lo[0]) * px_per_mm)))
    out_h = max(2, int(math.ceil((hi[1] - lo[1]) * px_per_mm)))
    xs = lo[0] + (np.arange(out_w) + 0.5) / px_per_mm
    ys = hi[1] - (np.arange(out_h) + 0.5) / px_per_mm
    gx, gy = np.meshgrid(xs, ys)
    plane_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    H_inv = np.linalg.inv(H)
    src = apply_homography(H_inv, plane_pts)
    sx = src[:, 0].reshape(out_h, out_w)
    sy = src[:, 1].reshape(out_h, out_w)
    out = np.zeros((out_h, out_w, 3), dtype=float)
    for ch in range(3):
        out[..., ch] = map_coordinates(
            img[..., ch], [sy, sx], order=1, mode="constant", cval=0.0)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
