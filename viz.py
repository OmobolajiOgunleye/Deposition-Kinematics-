"""
viz.py

Plots and GIF animations for the deposition-kinematics toolkit (matplotlib only).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402

from deposition import TrackingResult  # noqa: E402
from kinematics import PlanarArm  # noqa: E402
from toolpaths import Toolpath  # noqa: E402

ARM_COLOR = "#274c77"
TOOL_COLOR = "#e63946"


def _draw_arm(ax, arm: PlanarArm, q, color=ARM_COLOR, alpha=1.0, lw=4, ms=7, **kw):
    p = arm.joint_positions(q)
    ax.plot(p[:, 0], p[:, 1], "-o", color=color, lw=lw, ms=ms, alpha=alpha,
            mfc="white", mec=color, solid_capstyle="round", **kw)
    ax.plot(p[-1, 0], p[-1, 1], "o", color=TOOL_COLOR, ms=ms, alpha=alpha, zorder=6)



def plot_workspace(arm2: PlanarArm, arm3: PlanarArm, out: str) -> None:
    """Workspace, the two IK solutions, and a manipulability map."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=110)

    # (1) 2R workspace + both elbow solutions
    ax = axes[0]
    r_out = arm2.reach
    r_in = abs(arm2.L[0] - arm2.L[1])
    ax.add_patch(Wedge(arm2.base, r_out, 0, 360, width=r_out - r_in,
                       fc="#cfe3f5", ec="#6a9bc9", lw=1))
    target = np.array([230.0, 120.0])
    for elbow, ls, lab in (("up", "-", "elbow up (q2 > 0)"), ("down", "--", "elbow down (q2 < 0)")):
        q = arm2.ik(*target, elbow=elbow)
        p = arm2.joint_positions(q)
        ax.plot(p[:, 0], p[:, 1], ls, marker="o", lw=3, ms=7, mfc="white",
                label=f"{lab}: q = ({np.degrees(q[0]):.1f}°, {np.degrees(q[1]):.1f}°)")
    ax.plot(*target, "*", color=TOOL_COLOR, ms=16, zorder=7)
    ax.set_title("2R workspace and the two IK solutions")
    ax.set_aspect("equal")
    ax.set_xlim(-r_out - 20, r_out + 20)
    ax.set_ylim(-r_out - 20, r_out + 20)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)

    # (2) manipulability map w = L1 L2 |sin q2|
    ax = axes[1]
    L1, L2 = arm2.L
    g = np.linspace(-r_out - 10, r_out + 10, 400)
    X, Y = np.meshgrid(g, g)
    c2 = (X ** 2 + Y ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    W = np.where(np.abs(c2) <= 1, L1 * L2 * np.sqrt(np.clip(1 - c2 ** 2, 0, None)), np.nan)
    m = ax.pcolormesh(X, Y, W, shading="auto", cmap="viridis")
    fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04, label="manipulability (mm²)")
    ax.set_title("2R manipulability: zero on the workspace boundary")
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    # (3) 3R: same tool pose, two elbow solutions
    ax = axes[2]
    pose = np.array([230.0, 90.0, np.radians(35.0)])
    for elbow, ls in (("up", "-"), ("down", "--")):
        q = arm3.ik(*pose, elbow=elbow)
        p = arm3.joint_positions(q)
        ax.plot(p[:, 0], p[:, 1], ls, marker="o", lw=3, ms=7, mfc="white",
                label=f"elbow {elbow}: q = ({', '.join(f'{np.degrees(v):.0f}°' for v in q)})")
    ax.plot(pose[0], pose[1], "*", color=TOOL_COLOR, ms=16, zorder=7)
    ax.set_title("3R: same tool pose (x, y, φ = 35°), two solutions")
    ax.set_aspect("equal")
    ax.set_xlim(-30, arm3.reach + 20)
    ax.set_ylim(-arm3.reach * 0.7, arm3.reach * 0.7)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)



def plot_summary(arm: PlanarArm, path: Toolpath, res: TrackingResult, out: str, title: str) -> None:
    """Top view with arm snapshots, joint angles, joint speeds and manipulability."""
    fig = plt.figure(figsize=(13, 8.5), dpi=110)
    gs = fig.add_gridspec(2, 2)
    t_min = res.t / 60.0
    xyz = path.xyz

    # top view
    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], c=t_min, s=3, cmap="copper", zorder=3)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="time (min)")
    idx0 = np.flatnonzero(path.layer == 0)
    for f, a in zip((0.0, 0.25, 0.5, 0.75), (0.25, 0.4, 0.55, 0.8)):
        _draw_arm(ax, arm, res.q[idx0[int(f * (len(idx0) - 1))]], alpha=a)
    ax.plot(*arm.base, "s", color="k", ms=9)
    ax.set_aspect("equal")
    ax.set_xlim(-0.1 * arm.reach, arm.reach)
    ax.set_ylim(-0.55 * arm.reach, 0.55 * arm.reach)
    ax.set_title("Top view: toolpath and arm at four moments of layer 1")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(alpha=0.3)

    # joint angles
    ax = fig.add_subplot(gs[0, 1])
    for j in range(arm.n):
        ax.plot(t_min, np.degrees(res.q[:, j]), lw=1.4, label=f"q{j + 1}")
    ax.set_title("Joint angles")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("degrees")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.18 * (hi - lo))  # headroom so the legend never hides data
    ax.legend(ncol=arm.n, loc="upper center")
    ax.grid(alpha=0.3)

    # joint speeds
    ax = fig.add_subplot(gs[1, 0])
    for j in range(arm.n):
        ax.plot(t_min[1:], np.degrees(np.abs(res.qdot[:, j])), lw=1.2, label=f"|q̇{j + 1}|")
    ax.set_title(f"Joint speeds at {res.speed:g} mm/s travel speed")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("deg/s")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.18 * (hi - lo))
    ax.legend(ncol=arm.n, loc="upper center")
    ax.grid(alpha=0.3)

    # manipulability
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t_min, res.manip, color="#2a9d8f", lw=1.4)
    ax.set_title("Manipulability along the path (higher = further from singularity)")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("sqrt(det(J Jᵀ))")
    ax.set_ylim(bottom=0)
    ax.text(0.02, 0.06, f"max FK(IK(p)) position error: {res.err.max():.1e} mm",
            transform=ax.transAxes, fontsize=9, bbox=dict(fc="white", ec="0.7"))
    ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out)
    plt.close(fig)

def animate_deposition(
    arm: PlanarArm,
    path: Toolpath,
    res: TrackingResult,
    out: str,
    title: str = "",
    n_frames: int = 110,
    fps: int = 20,
    hold: int = 15,
) -> None:
    """GIF: arm (top view, left) and the part growing layer by layer (3-D, right)."""
    xyz = path.xyz
    n_layers = path.n_layers
    cols = plt.cm.copper(np.linspace(0.3, 1.0, n_layers))
    starts = np.array([np.flatnonzero(path.layer == k)[0] for k in range(n_layers)])
    ends = np.array([np.flatnonzero(path.layer == k)[-1] + 1 for k in range(n_layers)])

    fig = plt.figure(figsize=(10, 4.7), dpi=80)
    gs = fig.add_gridspec(1, 2)
    ax = fig.add_subplot(gs[0])
    ax3 = fig.add_subplot(gs[1], projection="3d")

    # top view
    R = arm.reach
    ax.set_xlim(-0.1 * R, R)
    ax.set_ylim(-0.55 * R, 0.55 * R)
    ax.set_aspect("equal")
    ax.set_title("Arm (top view)", fontsize=10)
    ax.set_xlabel("x (mm)", fontsize=8)
    ax.set_ylabel("y (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    th = np.linspace(-np.pi / 2.4, np.pi / 2.4, 100)
    ax.plot(arm.base[0] + R * np.cos(th), arm.base[1] + R * np.sin(th), ":", color="0.6", lw=1)
    ax.plot(xyz[:, 0], xyz[:, 1], color="0.8", lw=0.8, zorder=1)
    ax.plot(*arm.base, "s", color="k", ms=8, zorder=2)
    lines2 = [ax.plot([], [], color=cols[k], lw=2.2, zorder=3)[0] for k in range(n_layers)]
    arm_line, = ax.plot([], [], "-o", color=ARM_COLOR, lw=5, ms=8, mfc="white",
                        mec=ARM_COLOR, solid_capstyle="round", zorder=5)
    tip_pt, = ax.plot([], [], "o", color=TOOL_COLOR, ms=7, zorder=6)
    info = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top",
                   family="monospace", fontsize=8,
                   bbox=dict(fc="white", ec="0.7", alpha=0.9))

    # 3D view (zoomed on the part)
    cx = 0.5 * (xyz[:, 0].min() + xyz[:, 0].max())
    cy = 0.5 * (xyz[:, 1].min() + xyz[:, 1].max())
    half = 0.5 * max(np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1])) + 12
    zmax = xyz[:, 2].max() + 22
    ax3.set_xlim(cx - half, cx + half)
    ax3.set_ylim(cy - half, cy + half)
    ax3.set_zlim(0, zmax)
    ax3.set_box_aspect((2 * half, 2 * half, zmax))
    ax3.view_init(elev=24, azim=-58)
    ax3.set_title("Part, built layer by layer", fontsize=10)
    ax3.set_xlabel("x", fontsize=8)
    ax3.set_ylabel("y", fontsize=8)
    ax3.set_zlabel("z (mm)", fontsize=8)
    ax3.tick_params(labelsize=7)
    ax3.zaxis.set_major_locator(plt.MaxNLocator(4))
    lines3 =[ax3.plot([], [], [], color=cols[k], lw=2.4)[0] for k in range(n_layers)]
    torch, = ax3.plot([], [], [], "-", color=TOOL_COLOR, lw=3)
    tip3, = ax3.plot([], [], [], "o", color=TOOL_COLOR, ms=5)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    idxs = np.linspace(0, len(xyz) - 1, n_frames).astype(int)
    idxs = np.concatenate([idxs, np.full(hold, idxs[-1])])

    def update(frame):
        i = idxs[frame]
        for k in range(n_layers):
            m = max(0, min(i + 1, ends[k]) - starts[k])
            sl = slice(starts[k], starts[k] + m)
            lines2[k].set_data(xyz[sl, 0], xyz[sl, 1])
            lines3[k].set_data_3d(xyz[sl, 0], xyz[sl, 1], xyz[sl, 2])
        pts = arm.joint_positions(res.q[i])
        arm_line.set_data(pts[:, 0], pts[:, 1])
        tip_pt.set_data([xyz[i, 0]], [xyz[i, 1]])
        torch.set_data_3d([xyz[i, 0]] * 2, [xyz[i, 1]] * 2, [xyz[i, 2] + 14, xyz[i, 2]])
        tip3.set_data_3d([xyz[i, 0]], [xyz[i, 1]], [xyz[i, 2]])
        info.set_text(
            f"layer {path.layer[i] + 1:>2}/{n_layers}\n"
            f"t = {res.t[i] / 60:5.1f} min\n"
            f"z = {xyz[i, 2]:5.1f} mm"
        )
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(idxs), blit=False)
    anim.save(out, writer=animation.PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
