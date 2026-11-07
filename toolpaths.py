"""
toolpaths.py

Layer-by-layer deposition toolpaths (as used in wire-arc additive manufacturing,
WAAM, and directed energy deposition, DED).

Each path is built layer by layer. The tool moves in the x-y plane (done by the
robot arm) and a separate vertical Z axis raises the tool by one layer height
between layers.

Two layer strategies are provided:

 ``circle_layers``  - a thin-walled cylinder: the same circle every layer.
 ``spiral_layers``  - a filled disc: an Archimedean spiral per layer, alternating
  inward / outward so that consecutive layers join without a travel move.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Toolpath", "resample_polyline", "heading_from_xy", "circle_layers", "spiral_layers"]


@dataclass
class Toolpath:
    """A multi-layer deposition toolpath.

    Attributes
    
    xyz : (N, 3) tool positions in mm (z is the height of the Z axis)
    layer : (N,) integer layer index of every point (0-based)
    heading : (N,) in-plane travel direction in rad (unwrapped within each layer)
    layer_height : layer height in mm
    name : human readable name
    """

    xyz: np.ndarray
    layer: np.ndarray
    heading: np.ndarray
    layer_height: float
    name: str = "toolpath"

    @property
    def n_layers(self) -> int:
        return int(self.layer.max()) + 1

    @property
    def arclength(self) -> np.ndarray:
        """Cumulative 3-D path length (mm), including the Z lift between layers."""
        steps = np.linalg.norm(np.diff(self.xyz, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(steps)])


def resample_polyline(pts: np.ndarray, ds: float) -> np.ndarray:
    """Resample a polyline to (nearly) uniform spacing ``ds`` along its arc length.

    Uniform spacing means a constant travel speed turns into a constant time step,
    which is what a deposition process wants (constant bead cross-section).
    """
    pts = np.asarray(pts, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(round(s[-1] / ds)) + 1)
    s_new = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(s_new, s, pts[:, k]) for k in range(pts.shape[1])])


def heading_from_xy(xy: np.ndarray) -> np.ndarray:
    """In-plane travel direction (rad) from finite differences, unwrapped along the path."""
    g = np.gradient(np.asarray(xy, dtype=float), axis=0)
    return np.unwrap(np.arctan2(g[:, 1], g[:, 0]))


def _stack(layer_xy: list[np.ndarray], layer_height: float, z0: float, name: str) -> Toolpath:
    xyz, layer, heading = [], [], []
    for k, xy in enumerate(layer_xy):
        z = z0 + k * layer_height
        xyz.append(np.column_stack([xy, np.full(len(xy), z)]))
        layer.append(np.full(len(xy), k, dtype=int))
        heading.append(heading_from_xy(xy))
    return Toolpath(
        xyz=np.vstack(xyz),
        layer=np.concatenate(layer),
        heading=np.concatenate(heading),
        layer_height=layer_height,
        name=name,
    )


def circle_layers(
    center=(0.0, 0.0),
    radius: float = 50.0,
    n_layers: int = 10,
    layer_height: float = 2.5,
    ds: float = 2.0,
    z0: float = 0.0,
) -> Toolpath:
    """Thin-walled cylinder: one counter-clockwise circle per layer."""
    cx, cy = center
    theta = np.linspace(0.0, 2.0 * np.pi, 2000)
    ring = np.column_stack([cx + radius * np.cos(theta), cy + radius * np.sin(theta)])
    ring = resample_polyline(ring, ds)
    return _stack([ring.copy() for _ in range(n_layers)], layer_height, z0, "circle_layers")


def spiral_layers(
    center=(0.0, 0.0),
    r_out: float = 50.0,
    r_in: float = 5.0,
    bead_pitch: float = 4.0,
    n_layers: int = 4,
    layer_height: float = 2.5,
    ds: float = 2.0,
    z0: float = 0.0,
    alternate: bool = True,
) -> Toolpath:
    """Filled disc: Archimedean spiral r(t) = r_out - (pitch / 2 pi) t per layer.

    ``bead_pitch`` is the distance between neighbouring turns (bead spacing).
    With ``alternate=True`` even layers spiral inward and odd layers outward, so each
    layer starts where the previous one ended.
    """
    cx, cy = center
    t_max = 2.0 * np.pi * (r_out - r_in) / bead_pitch
    t = np.linspace(0.0, t_max, 6000)
    r = r_out - bead_pitch * t / (2.0 * np.pi)
    spiral = np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])
    spiral = resample_polyline(spiral, ds)
    layers = []
    for k in range(n_layers):
        layers.append(spiral[::-1].copy() if (alternate and k % 2 == 1) else spiral.copy())
    return _stack(layers, layer_height, z0, "spiral_layers")
