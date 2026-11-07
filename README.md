# Deposition Kinematics: a planar-arm toolkit for layer-by-layer toolpaths

A small, fully tested Python/numpy toolkit for the kinematics of 2-joint (2R) and
3-joint (3R) planar robot arms, applied to a **layer-by-layer deposition toolpath
simulation** of the kind used in wire-arc additive manufacturing (WAAM) and directed
energy deposition (DED).

It covers the foundations that multi-robot manufacturing cells build on:
homogeneous transforms, forward kinematics, the Jacobian, inverse kinematics
(analytic and numerical), singularities, and following a toolpath at a constant
travel speed.

| 3R arm, circular wall, tool kept tangent to the path | 2R arm, filled disc from alternating spiral layers |
|:---:|:---:|
| ![circle](outputs/circle_3dof.gif) | ![spiral](outputs/spiral_2dof.gif) |

## What is in the box

| Feature | Where |
|---|---|
| Homogeneous transforms (`rot_z`, `trans`, `invert_transform`, DH) | `kinematics.py` |
| Forward kinematics, frame chain, joint positions | `PlanarArm.fk`, `PlanarArm.frames` |
| Jacobian (position, and position + orientation) | `PlanarArm.jacobian` |
| Analytic inverse kinematics, both elbow branches (2R and 3R) | `PlanarArm.ik` |
| Numerical IK (damped least squares) | `PlanarArm.ik_numeric` |
| Manipulability and singularity analysis | `PlanarArm.manipulability` |
| Layered toolpaths: circle wall, alternating spiral fill | `toolpaths.py` |
| Constant-speed tracking, joint speeds, error metrics | `deposition.py` |
| Plots and GIF animations | `viz.py`, `run_demo.py` |
| 26 unit tests | `tests/` |

## The maths

Joint angles are relative: $q_i$ is the angle of link $i$ with respect to link $i-1$.
Link $i$ has length $L_i$, and $s_i = q_1 + \dots + q_i$.

**Homogeneous transforms.** Each link is a rotation about $z$ followed by a translation
along the rotated $x$ axis, and the tool frame is the product of the chain:

$$
T_{i-1}^{\,i} = R_z(q_i)\,\mathrm{Trans}_x(L_i), \qquad
T_{0}^{\,n} = T_{\text{base}} \prod_{i=1}^{n} T_{i-1}^{\,i}
$$

**Forward kinematics.** The tool pose is $(x, y, \varphi)$:

$$
x = x_b + \sum_{i=1}^{n} L_i \cos s_i, \qquad
y = y_b + \sum_{i=1}^{n} L_i \sin s_i, \qquad
\varphi = s_n
$$

**Jacobian.** Joint speeds map to tool velocity, $\dot{\mathbf p} = J \dot{\mathbf q}$, with

$$
\frac{\partial x}{\partial q_j} = -\sum_{i \ge j} L_i \sin s_i, \qquad
\frac{\partial y}{\partial q_j} = \sum_{i \ge j} L_i \cos s_i, \qquad
\frac{\partial \varphi}{\partial q_j} = 1
$$

**Analytic inverse kinematics (2R).** For a target $(x, y)$ relative to joint 1:

$$
\cos q_2 = \frac{x^2 + y^2 - L_1^2 - L_2^2}{2 L_1 L_2}, \qquad
q_2 = \pm\operatorname{atan2}\!\big(\sqrt{1-\cos^2 q_2},\ \cos q_2\big)
$$

$$
q_1 = \operatorname{atan2}(y, x) - \operatorname{atan2}\!\big(L_2 \sin q_2,\ L_1 + L_2 \cos q_2\big)
$$

The $\pm$ gives the two elbow branches ("up" is $q_2 > 0$). A target is reachable only if
$|L_1 - L_2| \le \sqrt{x^2+y^2} \le L_1 + L_2$; otherwise `UnreachableError` is raised.

**Analytic inverse kinematics (3R).** For a target pose $(x, y, \varphi)$, step back from
the tool tip along the tool direction to get the wrist point, solve it as a 2R problem,
and take what is left for the last joint:

$$
\mathbf p_w = \begin{bmatrix} x - L_3\cos\varphi \\ y - L_3\sin\varphi \end{bmatrix}, \qquad
q_3 = \varphi - q_1 - q_2
$$

**Numerical inverse kinematics.** Damped least squares, which stays well behaved near singularities:

$$
\Delta \mathbf q = J^{\mathsf T}\big(J J^{\mathsf T} + \lambda^2 I\big)^{-1}\mathbf e
$$

**Manipulability.** $w = \sqrt{\det(J J^{\mathsf T})}$. For the 2R arm this is
$L_1 L_2\,|\sin q_2|$, which is zero when the arm is fully stretched or folded back
(the singular configurations, visible in the workspace figure below).

**Toolpaths.** A circle wall repeats the same circle every layer. The spiral fill uses
an Archimedean spiral with bead spacing $p$, alternating inward and outward each layer
so one layer starts where the previous one ended:

$$
r(t) = r_{\text{out}} - \frac{p}{2\pi}\,t
$$

Paths are resampled to equal arc-length spacing, so a constant travel speed $v$ gives
time $t = s / v$ and joint speeds $\dot{\mathbf q} = \dfrac{d\mathbf q}{ds}\, v$.

![workspace](outputs/workspace_and_ik.png)

## Robot model used in the simulation

The arm moves the tool in the x-y plane. A separate vertical **Z axis** raises the tool by
one layer height between layers, like a SCARA-style cell or a planar arm on a lifting
column. So the full machine is 2R+Z or 3R+Z, and Z is not part of the arm IK.

* Arm lengths: 2R is 200 mm and 150 mm; 3R adds a 60 mm tool link.
* Part: centred 240 mm from the base, radius 60 mm, layer height 2.5 mm.
* Travel speed: 8 mm/s (adjustable with `--speed`). The numbers are illustrative,
  not a specific robot.

## Results

All numbers below come from `python run_demo.py`.

| Case | Layers | Points | Bead length | Time at 8 mm/s | Max joint speeds (deg/s) | Min manipulability | Max tracking error |
|---|---|---|---|---|---|---|---|
| Circle wall, 3R, tool tangent | 16 | 3,024 | 6,032 mm | 12.6 min | 4.2 / 5.2 / 12.9 | 20,910 | 1.7e-13 mm |
| Spiral fill, 2R | 5 | 7,005 | 13,996 mm | 29.2 min | 2.6 / 3.7 | 26,159 | 1.6e-13 mm |

* The tracking error is the distance between the planned point and the forward kinematics
  of the inverse-kinematics solution, so it confirms the IK and FK are consistent.
* Both paths stay far from singularities, so joint speeds are smooth.
* With the tool kept tangent to the path, joint 3 turns about one full revolution per layer
  and unwinds while the Z axis lifts to the next layer. That unwinding move is not counted
  in the joint-speed numbers.

![circle summary](outputs/summary_circle_3dof.png)
![spiral summary](outputs/summary_spiral_2dof.png)

## How to run

```bash
git clone https://github.com/<your-username>/deposition-kinematics.git
cd deposition-kinematics

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest -q                          # 26 tests
python run_demo.py                 # regenerate every figure and GIF into outputs/
```

Useful options:

```bash
python run_demo.py --no-gif                                # static figures only (fast)
python run_demo.py --path spiral --dof 3 --tool fixed      # a custom case
python run_demo.py --path circle --dof 2 --speed 5         # slower travel speed
```

Using the library directly:

```python
import numpy as np
from kinematics import PlanarArm

arm = PlanarArm([200.0, 150.0, 60.0])

q = arm.ik(230.0, 90.0, np.radians(35), elbow="up")   # analytic IK for a pose
print(arm.fk(q))                                      # -> [230.  90.   0.6109]
print(arm.jacobian(q, orientation=True))              # 3 x 3 Jacobian
print(arm.manipulability(q, orientation=True))        # distance from singularity
```

## Project structure

```
kinematics.py     transforms, PlanarArm (FK, Jacobian, IK, manipulability)
toolpaths.py      circle and spiral layer generators, resampling
deposition.py     path tracking, joint speeds, metrics
viz.py            figures and GIF animations
run_demo.py       command-line entry point
tests/            pytest suite
outputs/          generated figures, GIFs and metrics.json
```

## Scope and limitations

This is a kinematic simulation, kept deliberately simple.

* The arm is planar with a separate Z axis. It does not model a 6-axis industrial robot.
* There are no dynamics, collision checking or joint limits, and no process physics
  (heat input, bead geometry, interpass temperature).
* The demo parameters are illustrative, and the results are simulation outputs, not
  measurements from hardware.

## Next steps

* Port the same model to ROS 2: a URDF of the arm, TF2 frames, and visualisation in RViz2.
* Extend from planar arms to 6-axis industrial robots with MoveIt 2.
* Coordinate two arms on one part and check the shared workspace.
* Add sensor feedback, for example fusing noisy measurements to estimate bead height
  with a Kalman filter.
