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
    <img src="https://img.shields.io/badge/license-LGPL-green.svg" alt="LGPL">
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

Check out the [tutorial](doc/tutorial.md) and the [examples](https://github.com/TimSchneider42/franky/tree/master/examples) for an introduction.
The full documentation can be found at [https://timschneider42.github.io/franky/](https://timschneider42.github.io/franky/).

If you do not have a robot at hand, you can also try the [simulation](doc/tutorial.md#simulation) first.


## 🚀 Features

- **Control your Franka robot directly from Python in just a few lines!**
  No more endless hours setting up ROS, juggling packages, or untangling dependencies. Just `pip install` — no ROS at all.

- **[Multiple control modes](doc/tutorial.md#motion-types)**: [Cartesian position](doc/tutorial.md#cartesian-position-control), [Cartesian velocity](doc/tutorial.md#cartesian-velocity-control), [Joint position](doc/tutorial.md#joint-position-control), [Joint velocity](doc/tutorial.md#joint-velocity-control), and [Impedance control](doc/tutorial.md#impedance-control)
  franky uses [Ruckig](https://github.com/pantor/ruckig) to generate smooth, time-optimal trajectories while respecting velocity, acceleration, and jerk limits.

- **[Real-time control from Python and C++](doc/tutorial.md#real-time-motions)**
  Need to change the target while the robot’s moving? No problem. franky replans trajectories on the fly so that you can preempt motions anytime.

- **[Reactive behavior](doc/tutorial.md#-real-time-reactions)**
  Robots don’t always go according to plan. franky lets you define reactions to unexpected events—like contact with the environment — so you can change course in real-time.

- **[Motion and reaction callbacks](doc/tutorial.md#motion-callbacks)**
  Want to monitor what’s happening under the hood? Add callbacks to your motions and reactions. They won’t block the control thread and are super handy for debugging or logging.

- **Things are moving too fast? [Tune the robot's dynamics to your needs](doc/tutorial.md#-robot)**
  Adjust max velocity, acceleration, and jerk to match your setup or task. Fine control for smooth, safe operation.

- **Full Python access to the libfranka API**
  Want to tweak impedance, read the robot state, set force thresholds, or mess with the Jacobian? Go for it. If libfranka supports it, chances are franky does, too.

- **Scared to test code on the real system?**: [franky-sim](https://github.com/TimSchneider42/franky-sim) provides **[simulator support](doc/tutorial.md#simulation) for franky**! It is easy to install and use and serves as a drop-in replacement for the real robot.

## 📖 Python Quickstart Guide

Real-time kernel already installed and real-time permissions granted? Just install franky via

```bash
pip install franky-control
```

Otherwise, follow the [setup instructions](#setup) first.

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

If you are seeing server version mismatch errors, such as
```
franky.IncompatibleVersionException: libfranka: Incompatible library version (server version: 5, library version: 9)
```
then your Franka robot is either not on the most recent firmware version, or you are using the older Franka Panda model.
In any case, it's no big deal; just check [here](https://frankarobotics.github.io/docs/compatibility.html) which libfranka version you need and follow our [instructions](#installing-franky) to install the appropriate franky wheels.

## <a id="setup" /> ⚙️ Setup

To install franky, you have to follow three steps:

1. Ensure that you are using a real-time kernel
2. Ensure that the executing user has permission to run real-time applications
3. Install franky via pip or build it from source

### Installing a real-time kernel

In order for Franky to function properly, it requires the underlying OS to use a real-time kernel.
Otherwise, you might see `communication_constrains_violation` errors.

To check whether your system is currently using a real-time kernel, type `uname -a`.
You should see something like this:

```
$ uname -a
Linux [PCNAME] 5.15.0-1056-realtime #63-Ubuntu SMP PREEMPT_RT ...
```

If it does not say PREEMPT_RT, you are not currently running a real-time kernel.

There are multiple ways of installing a real-time kernel.
You can [build it from source](https://frankarobotics.github.io/docs/doc/libfranka/docs/real_time_kernel.html) or, if you are using Ubuntu, it can be [enabled through Ubuntu Pro](https://ubuntu.com/real-time).

### Allowing the executing user to run real-time applications

First, create a group `realtime` and add your user (or whoever is running franky) to this group:

```bash
sudo addgroup realtime
sudo usermod -a -G realtime $(whoami)
```

Afterward, add the following limits to the real-time group in /etc/security/limits.conf:

```
@realtime soft rtprio 99
@realtime soft priority 99
@realtime soft memlock 102400
@realtime hard rtprio 99
@realtime hard priority 99
@realtime hard memlock 102400
```

Log out and log in again to let the changes take effect.

To verify that the changes were applied, check if your user is in the `realtime` group:

```bash
$ groups
... realtime
```

If real-time is not listed in your groups, try rebooting.

### Installing franky

To start using franky with Python and libfranka *0.21.2*, just install it via

```bash
pip install franky-control
```

We also provide wheels for libfranka versions *0.7.1*, *0.8.0*, *0.9.2*, *0.12.1*, *0.13.3*,
*0.14.2*, *0.17.0*, and *0.21.2*.
They can be installed via

```bash
VERSION=0-9-2
wget https://github.com/TimSchneider42/franky/releases/latest/download/libfranka_${VERSION}_wheels.zip
unzip libfranka_${VERSION}_wheels.zip
pip install numpy websockets>=11
pip install --no-index --find-links=./dist franky-control
```

#### Development builds

If you need the latest features before they make it into an official release, we provide wheels of the current `master` branch in the rolling [dev release](https://github.com/TimSchneider42/franky/releases/tag/dev).
These wheels are rebuilt on every push to `master` and are provided for all supported libfranka versions.
They can be installed via the [package index](https://timschneider42.github.io/franky/whl/) by adding the `--pre` flag:

```bash
# You can replace the libfranka version by any of the supported versions denoted above
pip install --pre franky-control --extra-index-url "https://timschneider42.github.io/franky/whl/libfranka-0.21.2/"
```

Development builds are versioned as pre-releases of the next patch version, and their version indicates the commit and libfranka version they were built against: e.g., if the latest release is *1.1.4*, then *1.1.5.dev1234+g8cb09e5.libfranka.0.9.2* is a development build of commit `8cb09e5` on `master` for libfranka *0.9.2*.

### Using Docker

To use franky within Docker we provide a [Dockerfile](docker/run/Dockerfile) and
accompanying [docker-compose](docker-compose.yml) file.

```bash
git clone --recurse-submodules https://github.com/timschneider42/franky.git
cd franky/
docker compose build franky-run
```

To use another version of libfranka than the default (0.21.2), add a build argument:

```bash
docker compose build franky-run --build-arg LIBFRANKA_VERSION=0.9.2
```

To run the container:

```bash
docker compose run franky-run bash
```

The container requires access to the host machine's network *and* elevated user rights to allow the Docker user to set RT
capabilities of the processes run from within it.

### Can I use CUDA jointly with franky?

Yes. However, you need to set `IGNORE_PREEMPT_RT_PRESENCE=1` during the installation and all subsequent updates of the CUDA drivers on the real-time kernel.

First, make sure that you have rebooted your system after installing the real-time kernel.
Then, add `IGNORE_PREEMPT_RT_PRESENCE=1` to `/etc/environment`, call `export IGNORE_PREEMPT_RT_PRESENCE=1` to also set it in the current session, and follow the instructions of Nvidia to install CUDA on your system.

If you are on Ubuntu, you can also use [this](tools/install_cuda_realtime.bash) script to install CUDA on your real-time system:
```bash
# Download the script
wget https://raw.githubusercontent.com/timschneider42/franky/master/tools/install_cuda_realtime.bash

# Inspect the script to ensure it does what you expect

# Make it executable
chmod +x install_cuda_realtime.bash

# Execute the script
./install_cuda_realtime.bash
```

Alternatively, if you are a cowboy and do not care about security, you can also use this one-liner to directly call the script without checking it:
```bash
bash <(wget -qO- https://raw.githubusercontent.com/timschneider42/franky/master/tools/install_cuda_realtime.bash)
```

### Is your robot connected to a different machine?

No problem!
There are two projects which let you run franky remotely via RPC with minimal effort: [franky-remote](https://github.com/kvasios/franky-remote) and [net_franky](https://github.com/yblei/net_franky).

Please note that I’m not involved in the development of these projects, so I cannot take any liability for its use.
If you decide to use it, please ensure that you credit the developers of these projects for their work.

### Building franky

franky is based on [libfranka](https://github.com/frankarobotics/libfranka), [Eigen](https://eigen.tuxfamily.org) for
transformation calculations and [pybind11](https://github.com/pybind/pybind11) for the Python bindings.
As the Franka is sensitive to acceleration discontinuities, it requires jerk-constrained motion generation, for which
franky uses the [Ruckig](https://ruckig.com) community version for Online Trajectory Generation (OTG).

After installing the dependencies (the exact versions can be found [here](#-development)), you can build and install
franky via

```bash
git clone --recurse-submodules git@github.com:timschneider42/franky.git
cd franky
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
make install
```

To use franky, you can also include it as a subproject in your parent CMake via `add_subdirectory(franky)` and then
`target_link_libraries(<target> franky)`.

If you need only the Python module, you can install franky via

```bash
pip install .
```

Make sure that the built library `_franky.cpython-3**-****-linux-gnu.so` is in the Python path, e.g. by adjusting
`PYTHONPATH` accordingly.

#### Building franky with Docker

For building franky and its wheels, we provide another Docker container that can also be launched using docker-compose:

```bash
docker compose build franky-build
docker compose run --rm franky-build run-tests  # To run the tests
docker compose run --rm franky-build build-wheels  # To build wheels for all supported python versions
```

The built wheels are placed in `build/dist/` and can be installed via

```bash
pip install --no-index --find-links=./build/dist franky-control
```

## 📚 Tutorial

franky comes with both a C++ and Python API that differ only regarding real-time capability.
Check out the [tutorial](doc/tutorial.md) for a detailed introduction to geometry types, motion types, real-time reactions, asynchronous motion execution, gripper control, the Desk web interface API, and simulation.

## 🛠️ Development

franky is currently tested against the following versions

- libfranka >=0.7.1
- Eigen 3.4.0
- Pybind11 3.0.4
- POCO 1.12.5p2
- Pinocchio 3.4.0
- Ruckig 0.17.3
- Python >=3.7
- Catch2 2.13.8 (for testing only)

## 📜 License

For non-commercial applications, this software is licensed under the LGPL v3.0.
If you want to use franky within commercial applications or under a different license, please contact us for individual
agreements.

## 🔍 Differences to frankx

franky started originally as a fork of [frankx](https://github.com/pantor/frankx), though both codebase and
functionality differ substantially from frankx by now.
Aside from bug fixes and general performance improvements, franky provides the following new features/improvements:

* [Motions can be updated asynchronously.](doc/tutorial.md#-real-time-motions)
* [Reactions allow for the registration of callbacks instead of just printing to stdout when fired.](doc/tutorial.md#-real-time-reactions)
* [Motions allow for the registration of callbacks for profiling.](doc/tutorial.md#motion-callbacks)
* [The robot state is also available during control.](doc/tutorial.md#robot-state)
* A larger part of the libfranka API is exposed to python (e.g., `setCollisionBehavior`, `setJoinImpedance`, and
  `setCartesianImpedance`).
* Cartesian motion generation handles boundaries in Euler angles properly.
* [There is a new joint motion type that supports waypoints.](doc/tutorial.md#-motion-types)
* [The signature of `Affine` changed.](doc/tutorial.md#-geometry) `Affine` does not handle elbow positions anymore.
  Instead, a new class `RobotPose` stores both the end-effector pose and optionally the elbow position.
* The `MotionData` class does not exist anymore.
  Instead, reactions and other settings moved to `Motion`.
* [The `Measure` class allows for arithmetic operations.](doc/tutorial.md#-real-time-reactions)
* Exceptions caused by libfranka are raised properly instead of being printed to stdout.
* [We provide wheels for both Franka Research 3 and the older Franka Panda](#-setup)
* franky supports [joint velocity control](doc/tutorial.md#joint-velocity-control)
  and [cartesian velocity control](doc/tutorial.md#cartesian-velocity-control)
* The dynamics limits are not hard-coded anymore but can be [set for each robot instance](doc/tutorial.md#-robot).

## Contributing

If you wish to contribute to this project, you are welcome to create a pull request.
Please run the [pre-commit](https://pre-commit.com/) hooks before submitting your pull request.
To install the pre-commit hooks, run:

1. [Install pre-commit](https://pre-commit.com/#install)
2. Install the Git hooks by running `pre-commit install` or, alternatively, run `pre-commit run --all-files` manually.
