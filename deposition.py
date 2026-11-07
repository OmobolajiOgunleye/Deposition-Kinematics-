"""
deposition.py
=============

Ties the arm (``kinematics.PlanarArm``) to a layered toolpath (``toolpaths.Toolpath``):
solves inverse kinematics along the whole path, then derives time, joint speeds,
manipulability and tracking error at a constant deposition travel speed.

Robot model used in the demos: the planar arm moves the tool in x-y and a separate
vertical Z axis raises the tool by one layer height per layer (like a SCARA-style
cell, or a planar arm on a lifting column). Z is therefore not part of the arm IK.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kinematics import PlanarArm
from toolpaths import Toolpath

__all__ = ["TrackingResult", "solve_toolpath", "summarize"]


@dataclass
class TrackingResult:
    q: np.ndarray  # (N, n) joint angles, rad
    tip: np.ndarray  # (N, 3) forward-kinematics pose [x, y, phi]
    t: np.ndarray  # (N,) time stamps, s
    qdot: np.ndarray  # (N-1, n) joint speeds, rad/s (0 across layer changes)
    manip: np.ndarray  # (N,) manipulability
    err: np.ndarray  # (N,) position error |FK(IK(p)) - p|, mm
    speed: float  # deposition travel speed, mm/s
    elbow: str
    tool_mode: str


def solve_toolpath(
    arm: PlanarArm,
    path: Toolpath,
    speed: float = 8.0,
    tool_mode: str = "fixed",
    fixed_phi: float = 0.0,
    elbow: str = "up",
) -> TrackingResult:
    """Run inverse kinematics along ``path`` and derive time-based quantities.

    Parameters
    ----------
    speed : travel speed along the path in mm/s (typical WAAM values are a few mm/s)
    tool_mode : for a 3R arm, ``"fixed"`` keeps the tool direction constant at
        ``fixed_phi``; ``"tangent"`` keeps the last link aligned with the travel
        direction (like a leading sensor). Ignored for a 2R arm.

    Joint angles are unwrapped inside each layer. Between layers the wrist may
    unwind a full turn while the Z axis lifts; that move is not counted as joint
    speed.
    """
    n = arm.n
    N = len(path.xyz)
    if n == 2:
        phi_all = None
        use_or = False
    else:
        if tool_mode == "tangent":
            phi_all = path.heading
        elif tool_mode == "fixed":
            phi_all = np.full(N, float(fixed_phi))
        else:
            raise ValueError("tool_mode must be 'fixed' or 'tangent'")
        use_or = True

    q = np.empty((N, n))
    for k in np.unique(path.layer):
        idx = np.flatnonzero(path.layer == k)
        phi = None if phi_all is None else phi_all[idx]
        q[idx] = arm.track_path(path.xyz[idx, :2], phi, elbow)

    tip = np.array([arm.fk(qi) for qi in q])
    err = np.linalg.norm(tip[:, :2] - path.xyz[:, :2], axis=1)
    manip = np.array([arm.manipulability(qi, orientation=use_or) for qi in q])

    s = path.arclength
    t = s / speed
    qdot = np.diff(q, axis=0) / np.diff(s)[:, None] * speed
    qdot[path.layer[1:] != path.layer[:-1]] = 0.0

    return TrackingResult(
        q=q, tip=tip, t=t, qdot=qdot, manip=manip, err=err,
        speed=float(speed), elbow=elbow, tool_mode=tool_mode if n == 3 else "n/a",
    )


def summarize(arm: PlanarArm, path: Toolpath, res: TrackingResult) -> dict:
    """Key numbers for a run (printed by the demo and quoted in the README)."""
    steps = np.linalg.norm(np.diff(path.xyz[:, :2], axis=0), axis=1)
    same_layer = path.layer[1:] == path.layer[:-1]
    return {
        "path": path.name,
        "arm": f"{arm.n}R, L = {', '.join(f'{v:g}' for v in arm.L)} mm",
        "layers": path.n_layers,
        "points": len(path.xyz),
        "deposited_length_mm": float(steps[same_layer].sum()),
        "total_time_min": float(res.t[-1] / 60.0),
        "max_joint_speed_deg_s": [float(v) for v in np.degrees(np.abs(res.qdot)).max(axis=0)],
        "min_manipulability": float(res.manip.min()),
        "max_tracking_error_mm": float(res.err.max()),
    }
