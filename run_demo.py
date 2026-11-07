#!/usr/bin/env python3
"""
run_demo.py - generate every figure and animation of the project.

Examples
--------
    python run_demo.py                       # both default demos + workspace figure
    python run_demo.py --no-gif              # only static figures (fast)
    python run_demo.py --path spiral --dof 3 --tool fixed --speed 6
"""
from __future__ import annotations

import argparse
import json
import os
import time

from deposition import solve_toolpath, summarize
from kinematics import PlanarArm
from toolpaths import circle_layers, spiral_layers
from viz import animate_deposition, plot_summary, plot_workspace

# Demo geometry (mm). The part sits in the middle of the arm's workspace, away from
# singularities. These numbers are illustrative, not a specific robot.
ARM_2R = [200.0, 150.0]
ARM_3R = [200.0, 150.0, 60.0]
PART_CENTER = (240.0, 0.0)

DEFAULT_DEMOS = [("circle", 3, "tangent"), ("spiral", 2, "fixed")]


def build_path(kind: str):
    if kind == "circle":
        return circle_layers(center=PART_CENTER, radius=60.0, n_layers=16, layer_height=2.5, ds=2.0)
    if kind == "spiral":
        return spiral_layers(center=PART_CENTER, r_out=60.0, r_in=6.0, bead_pitch=4.0,
                             n_layers=5, layer_height=2.5, ds=2.0)
    raise ValueError(kind)


def run_one(kind: str, dof: int, tool: str, speed: float, out: str, gif: bool, frames: int) -> dict:
    arm = PlanarArm(ARM_3R if dof == 3 else ARM_2R)
    path = build_path(kind)
    res = solve_toolpath(arm, path, speed=speed, tool_mode=tool)
    stats = summarize(arm, path, res)

    tag = f"{kind}_{dof}dof"
    mode = f", tool {tool}" if dof == 3 else ""
    title = f"{kind.capitalize()} layers with a {dof}R arm{mode} - {speed:g} mm/s"
    plot_summary(arm, path, res, os.path.join(out, f"summary_{tag}.png"), title)
    if gif:
        t0 = time.time()
        animate_deposition(arm, path, res, os.path.join(out, f"{tag}.gif"), title=title, n_frames=frames)
        print(f"  animation written in {time.time() - t0:.0f} s")
    return {"tag": tag, **stats}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", choices=["circle", "spiral"], help="run a single custom case")
    ap.add_argument("--dof", type=int, choices=[2, 3], help="2R or 3R arm (custom case)")
    ap.add_argument("--tool", choices=["fixed", "tangent"], default=None,
                    help="3R tool direction: fixed or following the path tangent")
    ap.add_argument("--speed", type=float, default=8.0, help="travel speed in mm/s (default 8)")
    ap.add_argument("--out", default="outputs", help="output folder (default: outputs)")
    ap.add_argument("--frames", type=int, default=110, help="GIF frames (default 110)")
    ap.add_argument("--no-gif", action="store_true", help="skip the (slower) GIF animations")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.path:
        dof = args.dof or (3 if args.path == "circle" else 2)
        tool = args.tool or ("tangent" if args.path == "circle" else "fixed")
        demos = [(args.path, dof, tool)]
    else:
        demos = DEFAULT_DEMOS

    if not args.path:
        plot_workspace(PlanarArm(ARM_2R), PlanarArm(ARM_3R), os.path.join(args.out, "workspace_and_ik.png"))
        print("wrote workspace_and_ik.png")

    all_stats = []
    for kind, dof, tool in demos:
        print(f"running {kind} / {dof}R / tool={tool} ...")
        all_stats.append(run_one(kind, dof, tool, args.speed, args.out, not args.no_gif, args.frames))

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(all_stats, f, indent=2)

    print("\n=== results ===")
    for s in all_stats:
        speeds = ", ".join(f"{v:.1f}" for v in s["max_joint_speed_deg_s"])
        print(
            f"{s['tag']:>12}: {s['layers']} layers, {s['points']} points, "
            f"{s['deposited_length_mm']:.0f} mm of bead, {s['total_time_min']:.1f} min, "
            f"max joint speeds [{speeds}] deg/s, min manipulability {s['min_manipulability']:.0f}, "
            f"max error {s['max_tracking_error_mm']:.1e} mm"
        )


if __name__ == "__main__":
    main()
