"""Tests for the kinematics toolkit. Run with:  pytest -q"""
import numpy as np
import pytest

from deposition import solve_toolpath
from kinematics import (
    PlanarArm,
    UnreachableError,
    dh_transform,
    invert_transform,
    link_transform,
    rot_z,
    trans,
    wrap_angle,
)
from toolpaths import circle_layers, resample_polyline, spiral_layers

RNG = np.random.default_rng(0)
ARM2 = PlanarArm([200.0, 150.0])
ARM3 = PlanarArm([200.0, 150.0, 60.0])


# ------------------------------ transforms --------------------------------- #
def test_rot_z_is_orthonormal_and_composes():
    R = rot_z(0.7)[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(rot_z(0.3) @ rot_z(0.4), rot_z(0.7))


def test_invert_transform():
    T = trans(1.0, -2.0, 3.0) @ rot_z(0.9)
    assert np.allclose(T @ invert_transform(T), np.eye(4))
    assert np.allclose(invert_transform(T) @ T, np.eye(4))


def test_dh_matches_link_transform_for_planar_arm():
    assert np.allclose(dh_transform(0.8, 0.0, 120.0, 0.0), link_transform(0.8, 120.0))


def test_wrap_angle():
    assert np.isclose(wrap_angle(3 * np.pi / 2), -np.pi / 2)
    assert np.isclose(wrap_angle(-3 * np.pi / 2), np.pi / 2)


# --------------------------- forward kinematics ---------------------------- #
def test_fk_known_poses():
    assert np.allclose(ARM2.fk([0, 0])[:2], [350.0, 0.0])
    # first link straight up, second link bent -90 deg -> points along +x
    assert np.allclose(ARM2.fk([np.pi / 2, -np.pi / 2])[:2], [150.0, 200.0])
    # tool direction is the sum of joint angles
    assert np.isclose(ARM3.fk([0.2, 0.3, 0.4])[2], 0.9)


def test_fk_with_base_offset():
    arm = PlanarArm([100.0, 100.0], base_xy=(10.0, -5.0))
    assert np.allclose(arm.fk([0, 0])[:2], [210.0, -5.0])


def test_frame_chain_matches_fk():
    q = np.array([0.4, -0.7, 0.9])
    T = ARM3.frames(q)[-1]
    x, y, phi = ARM3.fk(q)
    assert np.allclose(T[:2, 3], [x, y])
    assert np.isclose(np.arctan2(T[1, 0], T[0, 0]), phi)


# -------------------------------- Jacobian --------------------------------- #
@pytest.mark.parametrize("arm", [ARM2, ARM3])
def test_jacobian_matches_finite_differences(arm):
    for _ in range(20):
        q = RNG.uniform(-np.pi, np.pi, arm.n)
        J = arm.jacobian(q, orientation=True)
        eps = 1e-6
        Jn = np.zeros_like(J)
        for j in range(arm.n):
            dq = np.zeros(arm.n)
            dq[j] = eps
            Jn[:, j] = (arm.fk(q + dq) - arm.fk(q - dq)) / (2 * eps)
        # phi is wrapped, so compare orientation row modulo 2*pi
        Jn[2] = wrap_angle(Jn[2] * 1.0)
        assert np.allclose(J[:2], Jn[:2], atol=1e-4)
        assert np.allclose(J[2], np.ones(arm.n))


def test_manipulability_2r_closed_form():
    for _ in range(20):
        q = RNG.uniform(-np.pi, np.pi, 2)
        expected = ARM2.L[0] * ARM2.L[1] * abs(np.sin(q[1]))
        assert np.isclose(ARM2.manipulability(q), expected)


def test_singularity_when_arm_is_stretched():
    assert ARM2.manipulability([0.3, 0.0]) < 1e-9
    assert ARM2.manipulability([0.3, np.pi]) < 1e-9


# ------------------------- inverse kinematics ------------------------------ #
@pytest.mark.parametrize("elbow", ["up", "down"])
def test_ik_2r_roundtrip(elbow):
    for _ in range(50):
        r = RNG.uniform(abs(ARM2.L[0] - ARM2.L[1]) + 5, ARM2.reach - 5)
        a = RNG.uniform(-np.pi, np.pi)
        target = np.array([r * np.cos(a), r * np.sin(a)])
        q = ARM2.ik(*target, elbow=elbow)
        assert np.allclose(ARM2.fk(q)[:2], target, atol=1e-9)
        assert (q[1] > 0) == (elbow == "up")


@pytest.mark.parametrize("elbow", ["up", "down"])
def test_ik_3r_roundtrip_position_and_orientation(elbow):
    count = 0
    while count < 50:
        q_true = RNG.uniform(-2.5, 2.5, 3)
        pose = ARM3.fk(q_true)
        try:
            q = ARM3.ik(pose[0], pose[1], pose[2], elbow=elbow)
        except UnreachableError:
            continue
        got = ARM3.fk(q)
        assert np.allclose(got[:2], pose[:2], atol=1e-8)
        assert np.isclose(wrap_angle(got[2] - pose[2]), 0.0, atol=1e-8)
        count += 1


def test_ik_unreachable_raises():
    with pytest.raises(UnreachableError):
        ARM2.ik(400.0, 0.0)  # beyond reach (350)
    with pytest.raises(UnreachableError):
        ARM2.ik(10.0, 0.0)  # inside the inner dead zone (|L1 - L2| = 50)


def test_ik_3r_requires_phi():
    with pytest.raises(ValueError):
        ARM3.ik(200.0, 50.0)


def test_numeric_ik_converges_to_target():
    for _ in range(20):
        q_true = RNG.uniform(-1.2, 1.2, 3)
        pose = ARM3.fk(q_true)
        q0 = q_true + RNG.normal(0, 0.3, 3)
        q, ok, _ = ARM3.ik_numeric(pose, q0, use_orientation=True, tol=1e-8)
        assert ok
        got = ARM3.fk(q)
        assert np.allclose(got[:2], pose[:2], atol=1e-6)
        assert np.isclose(wrap_angle(got[2] - pose[2]), 0.0, atol=1e-6)


def test_numeric_ik_matches_analytic_position():
    target = np.array([230.0, 80.0])
    q_an = ARM2.ik(*target, elbow="up")
    q_num, ok, _ = ARM2.ik_numeric(target, q_an + 0.2)
    assert ok
    assert np.allclose(ARM2.fk(q_num)[:2], target, atol=1e-6)


# ------------------------------- toolpaths --------------------------------- #
def test_resample_gives_uniform_spacing():
    t = np.linspace(0, 2 * np.pi, 400)
    pts = np.column_stack([50 * np.cos(t), 50 * np.sin(t)])
    out = resample_polyline(pts, 2.0)
    d = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert d.std() / d.mean() < 0.01
    assert abs(d.mean() - 2.0) < 0.1


def test_circle_layers_structure():
    p = circle_layers(center=(240, 0), radius=60, n_layers=5, layer_height=2.5)
    assert p.n_layers == 5
    r = np.hypot(p.xyz[:, 0] - 240, p.xyz[:, 1])
    assert np.allclose(r, 60, atol=0.05)
    z_levels = np.unique(np.round(p.xyz[:, 2], 6))
    assert np.allclose(z_levels, np.arange(5) * 2.5)


def test_spiral_layers_alternate_direction():
    p = spiral_layers(center=(240, 0), r_out=60, r_in=6, n_layers=2)
    r = np.hypot(p.xyz[:, 0] - 240, p.xyz[:, 1])
    r0 = r[p.layer == 0]
    r1 = r[p.layer == 1]
    assert r0[0] > r0[-1] + 40  # layer 0 spirals inward
    assert r1[0] < r1[-1] - 40  # layer 1 spirals outward
    assert np.isclose(r0[-1], r1[0], atol=0.5)  # joined without a travel move


# --------------------------- whole-path tracking --------------------------- #
@pytest.mark.parametrize(
    "arm,mode", [(ARM2, "fixed"), (ARM3, "fixed"), (ARM3, "tangent")]
)
def test_tracking_circle_is_exact_and_smooth(arm, mode):
    path = circle_layers(center=(240, 0), radius=60, n_layers=3)
    res = solve_toolpath(arm, path, speed=8.0, tool_mode=mode)
    assert res.err.max() < 1e-8
    assert res.manip.min() > 1e3  # far from singularities
    for k in range(3):
        qk = res.q[path.layer == k]
        assert np.abs(np.diff(qk, axis=0)).max() < 0.1  # no joint jumps within a layer


def test_jacobian_maps_joint_motion_to_tool_velocity():
    path = circle_layers(center=(240, 0), radius=60, n_layers=1)
    res = solve_toolpath(ARM3, path, tool_mode="tangent")
    for i in range(0, len(path.xyz) - 1, 15):
        dq = res.q[i + 1] - res.q[i]
        q_mid = 0.5 * (res.q[i] + res.q[i + 1])
        dp = ARM3.jacobian(q_mid, orientation=True) @ dq
        actual = res.tip[i + 1] - res.tip[i]
        actual[2] = wrap_angle(actual[2])
        assert np.allclose(dp[:2], actual[:2], atol=1e-3)
    # tool speed along the path equals the commanded travel speed
    v = np.linalg.norm(np.diff(path.xyz[:, :2], axis=0), axis=1) / np.diff(res.t)
    same = path.layer[1:] == path.layer[:-1]
    assert np.allclose(v[same], 8.0, rtol=1e-6)
