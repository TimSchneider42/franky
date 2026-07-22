<div align="center">
  <img width="340" src="https://raw.githubusercontent.com/timschneider42/franky/master/doc/logo.svg?sanitize=true">
  <h3 align="center">
    High-Level Control Library for Franka Robots with Python and C++ Support
  </h3>
</div>
<p align="center">
  <a href="https://github.com/timschneider42/franky/actions">
    <img src="https://github.com/timschneider42/franky/workflows/CI/badge.svg" alt="CI">
  </a>

  <a href="https://github.com/timschneider42/franky/actions">
    <img src="https://github.com/timschneider42/franky/workflows/Publish/badge.svg" alt="Publish">
  </a>

  <a href="https://github.com/timschneider42/franky/issues">
    <img src="https://img.shields.io/github/issues/timschneider42/franky.svg" alt="Issues">
  </a>

  <a href="https://github.com/timschneider42/franky/releases">
    <img src="https://img.shields.io/github/v/release/timschneider42/franky.svg?include_prereleases&sort=semver" alt="Releases">
  </a>

  <a href="https://github.com/timschneider42/franky/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT">
  </a>
</p>

franky is a high-level control library for Franka robots, offering Python and C++ support.
By providing a high-level control interface, franky eliminates the need for strict real-time programming at 1 kHz,
making control from non-real-time environments, such as Python programs, feasible.
Instead of relying on low-level control commands, franky expects high-level position or velocity targets and
uses [Ruckig](https://github.com/pantor/ruckig) to plan time-optimal trajectories in real-time.

Although Python does not provide real-time guarantees, franky strives to maintain as much real-time control as possible.
Motions can be preempted at any moment, prompting franky to re-plan trajectories on the fly.
To handle unforeseen situations—such as unexpected contact with the environment — franky includes a reaction system that
allows for updating motion commands dynamically.
Furthermore, most non-real-time functionality of [libfranka](https://frankarobotics.github.io/docs/doc/libfranka/docs/index.html), such as
Gripper control is made directly available in Python.

Check out the [documentation](https://timschneider42.github.io/franky/) and
the [examples](https://github.com/TimSchneider42/franky/tree/master/examples) for an introduction.

If you do not have a robot at hand, you can also try
the [simulation](https://timschneider42.github.io/franky/guide/simulation.html) first.

## 🚀 Features

- **Control your Franka robot directly from Python in just a few lines!**
  No more endless hours setting up ROS, juggling packages, or untangling dependencies. Just `pip install` — no ROS at all.

- **[Multiple control modes](https://timschneider42.github.io/franky/guide/motion-types.html)**: [Cartesian position](https://timschneider42.github.io/franky/guide/motion-types.html#cartesian-position-control), [Cartesian velocity](https://timschneider42.github.io/franky/guide/motion-types.html#cartesian-velocity-control), [Joint position](https://timschneider42.github.io/franky/guide/motion-types.html#joint-position-control), [Joint velocity](https://timschneider42.github.io/franky/guide/motion-types.html#joint-velocity-control), [Impedance control](https://timschneider42.github.io/franky/guide/impedance-control.html), and [Torque control](https://timschneider42.github.io/franky/guide/torque-control.html)
  franky uses [Ruckig](https://github.com/pantor/ruckig) to generate smooth, time-optimal trajectories while respecting velocity, acceleration, and jerk limits.

- **[Real-time control from Python and C++](https://timschneider42.github.io/franky/guide/realtime-motions.html)**
  Need to change the target while the robot’s moving? No problem. franky replans trajectories on the fly so that you can preempt motions anytime.

- **[Reactive behavior](https://timschneider42.github.io/franky/guide/reactions.html)**
  Robots don’t always go according to plan. franky lets you define reactions to unexpected events—like contact with the environment — so you can change course in real-time.

- **[Motion and reaction callbacks](https://timschneider42.github.io/franky/guide/executing-motions.html#motion-callbacks)**
  Want to monitor what’s happening under the hood? Add callbacks to your motions and reactions. They won’t block the control thread and are super handy for debugging or logging.

- **Things are moving too fast? [Tune the robot's dynamics to your needs](https://timschneider42.github.io/franky/guide/robot.html)**
  Adjust max velocity, acceleration, and jerk to match your setup or task. Fine control for smooth, safe operation.

- **Full Python access to the libfranka API**
  Want to tweak impedance, read the robot state, set force thresholds, or mess with the Jacobian? Go for it. If libfranka supports it, chances are franky does, too.

- **Scared to test code on the real system?**: [franky-sim](https://github.com/TimSchneider42/franky-sim) provides **[simulator support](https://timschneider42.github.io/franky/guide/simulation.html) for franky**! It is easy to install and use and serves as a drop-in replacement for the real robot.

## 📖 Python Quickstart Guide

Real-time kernel already installed and real-time permissions granted? Just install franky via

```bash
pip install franky-control
```

Otherwise, follow the [installation instructions](https://timschneider42.github.io/franky/getting-started/installation.html) first.

We also provide development builds of the latest `master` (stable) and `dev` (unstable) branches:

```bash
# Replace 10 with your robot server version
# Development (stable): latest master branch
pip install --pre franky-control --extra-index-url "https://timschneider42.github.io/franky/whl/by-robot-server-version/10/"
# Development (unstable): latest dev branch
pip install --pre franky-control --extra-index-url "https://timschneider42.github.io/franky/branches/dev/whl/by-robot-server-version/10/"
```

Now we are already ready to go!
Unlock the brakes in the web interface, activate FCI, and start coding:

```python
from franky import *

robot = Robot("10.90.90.1")  # Replace this with your robot's IP

# Let's start slow (this lets the robot use a maximum of 5% of its velocity, acceleration, and jerk limits)
robot.relative_dynamics_factor = 0.05

# Move the robot 20cm along the relative X-axis of its end-effector
motion = CartesianMotion(Affine([0.2, 0.0, 0.0]), ReferenceType.Relative)
robot.move(motion)
```

If you are seeing server version mismatch errors, your robot is likely running an older firmware or is an older Franka
Panda model — see
the [FAQ](https://timschneider42.github.io/franky/getting-started/faq.html#i-am-seeing-server-version-mismatch-errors-what-now)
for how to install matching wheels.

## 📚 Documentation

The full documentation is available at [https://timschneider42.github.io/franky/](https://timschneider42.github.io/franky/):

- [Installation](https://timschneider42.github.io/franky/getting-started/installation.html) — real-time kernel setup, real-time permissions, wheels for older libfranka versions, and [Docker](https://timschneider42.github.io/franky/getting-started/docker.html)
- [Guide](https://timschneider42.github.io/franky/guide/index.html) — geometry types, motion types, real-time reactions, asynchronous motion execution, gripper control, the Desk web interface API, and simulation
- [FAQ](https://timschneider42.github.io/franky/getting-started/faq.html) — CUDA on real-time kernels, remote robot setups, and more
- [Development](https://timschneider42.github.io/franky/development.html) — building from source, tested dependency versions, differences to frankx, and contributing

## 📜 License

Starting with version 2.0.0, franky is licensed under the [MIT License](LICENSE).
Versions prior to 2.0.0 were released under the LGPL v3.0 and remain available under that license.

The Python docstrings of the `franka.*` entities ([`python/docstrings.hpp`](python/docstrings.hpp)) are extracted from
the headers of [libfranka](https://github.com/frankaemika/libfranka), Copyright (c) Franka Robotics GmbH, which is
licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).

## Contributing

If you wish to contribute to this project, you are welcome to create a pull request.
Please check out
the [contributing guidelines](https://timschneider42.github.io/franky/development.html#contributing) first.
