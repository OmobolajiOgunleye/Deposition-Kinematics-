"""
kinematics.py
=============

Kinematics toolkit for planar serial arms with 2 or 3 revolute joints (2R / 3R).

Everything is plain numpy so it is easy to read, test and port later
(for example to a ROS 2 node or a URDF-based setup).

Conventions
-----------
* Units: millimetres and radians.
* Joint angle q_i is measured relative to the previous link (relative angles).
  Link i has length L_i, and the arm moves in the x-y plane.
* Poses are (x, y, phi) where phi = q_1 + ... + q_n is the absolute heading
  of the last link (the "tool direction").
* Elbow branch: ``elbow="up"`` means q_2 > 0, ``elbow="down"`` means q_2 < 0.

Contents
--------
* Homogeneous transforms: ``rot_z``, ``trans``, ``link_transform``,
  ``dh_transform``, ``invert_transform``
* ``PlanarArm`` with forward kinematics, Jacobian, analytic inverse kinematics,
  numerical (damped least squares) inverse kinematics, manipulability and a
  helper that tracks a whole path.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "wrap_angle",
    "rot_z",
    "trans",
    "invert_transform",
    "link_transform",
    "dh_transform",
    "UnreachableError",
    "PlanarArm",
]


# --------------------------------------------------------------------------- #
# Homogeneous transforms (4x4, SE(3) restricted to rotation about z)
# --------------------------------------------------------------------------- #
def wrap_angle(a):
    """Wrap an angle (or array of angles) to the interval [-pi, pi)."""
    return (np.asarray(a, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def rot_z(theta: float) -> np.ndarray:
    """Homogeneous transform: rotation by ``theta`` about the z axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def trans(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    """Homogeneous transform: pure translation."""
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid-body homogeneous transform (uses R^T, not a generic inverse)."""
    R = T[:3, :3]
    p = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


def link_transform(theta: float, length: float) -> np.ndarray:
    """Transform from one joint frame to the next for a planar link.

    T = Rz(theta) * Tx(length)
    """
    return rot_z(theta) @ trans(length, 0.0, 0.0)


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Standard Denavit-Hartenberg transform (included for reference/comparison).

    For a planar arm (d = 0, alpha = 0) this equals ``link_transform(theta, a)``.
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


# --------------------------------------------------------------------------- #
# Planar serial arm
# --------------------------------------------------------------------------- #
class UnreachableError(ValueError):
    """Raised when a target point is outside the arm's reachable workspace."""


class PlanarArm:
    """Planar serial arm with revolute joints.

    Parameters
    ----------
    link_lengths : sequence of float
        Link lengths in mm. Length 2 gives a 2R arm, length 3 a 3R arm.
    base_xy : (float, float)
        Position of joint 1 in the world frame (mm).
    """

    def __init__(self, link_lengths, base_xy=(0.0, 0.0)):
        L = np.asarray(link_lengths, dtype=float)
        if L.ndim != 1 or L.size < 2:
            raise ValueError("link_lengths must be a 1-D sequence with at least 2 entries")
        if np.any(L <= 0):
            raise ValueError("link lengths must be positive")
        self.L = L
        self.n = int(L.size)
        self.base = np.asarray(base_xy, dtype=float)

    # ------------------------------------------------------------------ #
    @property
    def reach(self) -> float:
        """Maximum reach of the fully stretched arm (mm)."""
        return float(self.L.sum())

    def _q(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape != (self.n,):
            raise ValueError(f"expected {self.n} joint angles, got shape {q.shape}")
        return q

    # ------------------------------------------------------------------ #
    # Forward kinematics
    # ------------------------------------------------------------------ #
    def frames(self, q) -> list[np.ndarray]:
        """Chain of 4x4 world-frame transforms: base, frame 1, ..., tool frame."""
        q = self._q(q)
        T = trans(self.base[0], self.base[1], 0.0)
        chain = [T]
        for theta, length in zip(q, self.L):
            T = T @ link_transform(theta, length)
            chain.append(T)
        return chain

    def joint_positions(self, q) -> np.ndarray:
        """(n+1, 2) array: base, joint 2, ..., tool tip (x, y) in the world frame."""
        return np.array([T[:2, 3] for T in self.frames(q)])

    def fk(self, q) -> np.ndarray:
        """Forward kinematics. Returns the tool pose [x, y, phi]."""
        T = self.frames(q)[-1]
        return np.array([T[0, 3], T[1, 3], np.arctan2(T[1, 0], T[0, 0])])

    # ------------------------------------------------------------------ #
    # Jacobian
    # ------------------------------------------------------------------ #
    def jacobian(self, q, orientation: bool = False) -> np.ndarray:
        """Geometric Jacobian of the tool.

        Position part (2 x n), with s_i = q_1 + ... + q_i:

            dx/dq_j = -sum_{i>=j} L_i sin(s_i)
            dy/dq_j = +sum_{i>=j} L_i cos(s_i)

        If ``orientation`` is True a third row of ones is appended
        (d phi / d q_j = 1), giving a 3 x n matrix.
        """
        q = self._q(q)
        s = np.cumsum(q)
        dx = -(self.L * np.sin(s))
        dy = self.L * np.cos(s)
        Jx = np.cumsum(dx[::-1])[::-1]
        Jy = np.cumsum(dy[::-1])[::-1]
        J = np.vstack([Jx, Jy])
        if orientation:
            J = np.vstack([J, np.ones(self.n)])
        return J

    def manipulability(self, q, orientation: bool = False) -> float:
        """Yoshikawa manipulability w = sqrt(det(J J^T)). Goes to 0 at singularities."""
        J = self.jacobian(q, orientation=orientation)
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    # ------------------------------------------------------------------ #
    # Inverse kinematics (analytic)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ik_two_link(dx: float, dy: float, l1: float, l2: float, elbow: str):
        """Closed-form 2R inverse kinematics for a target relative to joint 1."""
        c2 = (dx * dx + dy * dy - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
        if abs(c2) > 1.0 + 1e-9:
            raise UnreachableError(
                f"target at distance {np.hypot(dx, dy):.3f} mm is outside the workspace "
                f"[{abs(l1 - l2):.3f}, {l1 + l2:.3f}] mm"
            )
        c2 = float(np.clip(c2, -1.0, 1.0))
        s2 = np.sqrt(1.0 - c2 * c2) * (1.0 if elbow == "up" else -1.0)
        q2 = np.arctan2(s2, c2)
        q1 = np.arctan2(dy, dx) - np.arctan2(l2 * s2, l1 + l2 * c2)
        return q1, q2

    def ik(self, x: float, y: float, phi: float | None = None, elbow: str = "up") -> np.ndarray:
        """Analytic inverse kinematics.

        * 2R arm: solves for the position (x, y); ``phi`` is ignored
          (the tool direction is determined by the solution).
        * 3R arm: solves for the full pose (x, y, phi). The wrist point
          (tool tip minus L3 along phi) is solved as a 2R problem, then
          q3 = phi - q1 - q2.

        Raises ``UnreachableError`` if the target is outside the workspace.
        """
        if elbow not in ("up", "down"):
            raise ValueError("elbow must be 'up' or 'down'")
        dx = float(x) - self.base[0]
        dy = float(y) - self.base[1]

        if self.n == 2:
            q1, q2 = self._ik_two_link(dx, dy, self.L[0], self.L[1], elbow)
            return wrap_angle(np.array([q1, q2]))

        if self.n == 3:
            if phi is None:
                raise ValueError("a 3R arm needs a tool direction phi for analytic IK")
            wx = dx - self.L[2] * np.cos(phi)
            wy = dy - self.L[2] * np.sin(phi)
            q1, q2 = self._ik_two_link(wx, wy, self.L[0], self.L[1], elbow)
            q3 = phi - q1 - q2
            return wrap_angle(np.array([q1, q2, q3]))

        raise NotImplementedError("analytic IK is implemented for 2R and 3R arms only")

    # ------------------------------------------------------------------ #
    # Inverse kinematics (numerical, damped least squares)
    # ------------------------------------------------------------------ #
    def ik_numeric(
        self,
        target,
        q0,
        use_orientation: bool = False,
        tol: float = 1e-9,
        max_iter: int = 200,
        damping: float = 1e-3,
        max_step: float = 0.3,
    ):
        """Damped least squares (Levenberg-Marquardt style) inverse kinematics.

        Update rule:  dq = J^T (J J^T + lambda^2 I)^-1 e

        Parameters
        ----------
        target : array-like
            (x, y) or (x, y, phi) if ``use_orientation`` is True.
        q0 : array-like
            Initial guess (also selects the elbow branch it converges to).

        Returns
        -------
        q : ndarray, converged : bool, iterations : int
        """
        if use_orientation and self.n < 3:
            raise ValueError("orientation task needs at least 3 joints")
        target = np.asarray(target, dtype=float)
        q = self._q(q0).copy()
        for it in range(max_iter):
            pose = self.fk(q)
            e = target[:2] - pose[:2]
            if use_orientation:
                e = np.append(e, wrap_angle(target[2] - pose[2]))
            if np.linalg.norm(e) < tol:
                return q, True, it
            J = self.jacobian(q, orientation=use_orientation)
            JJt = J @ J.T + (damping ** 2) * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJt, e)
            norm = np.linalg.norm(dq)
            if norm > max_step:
                dq *= max_step / norm
            q = q + dq
        pose = self.fk(q)
        e = target[:2] - pose[:2]
        if use_orientation:
            e = np.append(e, wrap_angle(target[2] - pose[2]))
        return q, bool(np.linalg.norm(e) < tol), max_iter

    # ------------------------------------------------------------------ #
    # Path tracking
    # ------------------------------------------------------------------ #
    def track_path(self, xy, phi=None, elbow: str = "up") -> np.ndarray:
        """Solve IK along a sequence of points, staying on one elbow branch.

        Joint angles are unwrapped so they are continuous along the path.

        Parameters
        ----------
        xy : (N, 2) array of tool positions
        phi : (N,) array of tool directions (required for 3R, ignored for 2R)
        """
        xy = np.asarray(xy, dtype=float)
        q = np.empty((len(xy), self.n))
        for i, (x, y) in enumerate(xy):
            q[i] = self.ik(x, y, None if phi is None else phi[i], elbow)
        return np.unwrap(q, axis=0)
