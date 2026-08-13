# 🛠️ Development

## Building franky from Source

franky is based on [libfranka](https://github.com/frankarobotics/libfranka), [Eigen](https://eigen.tuxfamily.org) for
transformation calculations and [pybind11](https://github.com/pybind/pybind11) for the Python bindings.
As the Franka is sensitive to acceleration discontinuities, it requires jerk-constrained motion generation, for which
franky uses the [Ruckig](https://ruckig.com) community version for Online Trajectory Generation (OTG).

After installing the dependencies (the exact versions can be found [below](#tested-dependency-versions)), you can build
and install franky via

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

### Building franky with Docker

For building franky and its wheels, we provide another Docker container that can also be launched using
docker-compose:

```bash
docker compose build franky-build
docker compose run --rm franky-build run-tests  # To run the tests
docker compose run --rm franky-build build-wheels  # To build wheels for all supported python versions
```

The built wheels are placed in `build/dist/` and can be installed via

```bash
pip install --no-index --find-links=./build/dist franky-control
```

## Tested Dependency Versions

franky is currently tested against the following versions

- libfranka >=0.7.1
- Eigen 3.4.0
- Pybind11 3.0.4
- POCO 1.12.5p2
- Pinocchio 3.4.0
- Ruckig 0.17.3
- Python >=3.7
- Catch2 2.13.8 (for testing only)

## Regenerating the Python Docstrings

The docstrings of the Python bindings are extracted from the doxygen comments in the C++ headers
with [pybind11_mkdoc](https://github.com/pybind/pybind11_mkdoc) and are checked in as
`python/docstrings.hpp`, so regular builds do not depend on pybind11_mkdoc. After changing doc
comments in the headers, regenerate this file with the `franky-docstrings` CMake target, which uses
the include directories and version defines of the configured build:

```bash
sudo apt-get install libclang-dev
pip install pybind11_mkdoc "clang~=14.0"  # the clang package version should match libclang
cmake --build build --target franky-docstrings
```

## Building the Documentation

The documentation is built with [Sphinx](https://www.sphinx-doc.org/).
Generating the Python API reference requires franky to be importable, so install it (e.g. a wheel or a source build)
first:

```bash
pip install franky-control  # or: pip install .
pip install -r doc/requirements.txt
VERSION=$(cat VERSION) sphinx-build -b html doc doc/_build/html
```

## Differences to frankx

franky started originally as a fork of [frankx](https://github.com/pantor/frankx), though both codebase and
functionality differ substantially from frankx by now.
Aside from bug fixes and general performance improvements, franky provides the following new features/improvements:

* [Motions can be updated asynchronously.](guide/realtime-motions.md)
* [Reactions allow for the registration of callbacks instead of just printing to stdout when fired.](guide/reactions.md)
* [Motions allow for the registration of callbacks for profiling.](guide/executing-motions.md#motion-callbacks)
* [The robot state is also available during control.](guide/robot.md#robot-state)
* A larger part of the libfranka API is exposed to python (e.g., `setCollisionBehavior`, `setJointImpedance`, and
  `setCartesianImpedance`).
* Cartesian motion generation handles boundaries in Euler angles properly.
* [Motions in both joint and Cartesian space support waypoints with target velocities.](guide/trajectory-control.md)
* franky supports [joint velocity control](guide/trajectory-control.md#joint-velocity-control)
  and [cartesian velocity control](guide/trajectory-control.md#cartesian-velocity-control).
* [franky provides client-side joint and Cartesian impedance controllers](guide/impedance-control.md), including
  tracking variants whose references and gains can be updated in real time.
* [Joint torques can be commanded directly](guide/torque-control.md) with `SimpleTorqueMotion`.
* [The robot's kinematics model is accessible](guide/robot.md#robot-state) for computing forward kinematics and
  Jacobians, and the robot's URDF model can be retrieved.
* [The Franka Desk API can be accessed programmatically](guide/desk.md), e.g., to unlock the brakes, enable the FCI,
  or run the safety self-test.
* [franky can be used with the franky-sim simulator](guide/simulation.md), a drop-in replacement for the real robot.
* [The signature of `Affine` changed.](guide/geometry.md) `Affine` does not handle elbow positions anymore.
  Instead, a new class `RobotPose` stores both the end-effector pose and optionally the elbow position.
* The `MotionData` class does not exist anymore.
  Instead, reactions and other settings moved to `Motion`.
* [The `Measure` class allows for arithmetic operations.](guide/reactions.md)
* Exceptions caused by libfranka are raised properly instead of being printed to stdout.
* [We provide wheels for a wide range of libfranka versions](getting-started/installation.md), supporting both the
  Franka Research 3 and the older Franka Panda.
* The dynamics limits are not hard-coded anymore but can be [set for each robot instance](guide/robot.md).

## Contributing

If you wish to contribute to this project, you are welcome to create a pull request.
Please run the [pre-commit](https://pre-commit.com/) hooks before submitting your pull request.
To install the pre-commit hooks, run:

1. [Install pre-commit](https://pre-commit.com/#install)
2. Install the Git hooks by running `pre-commit install` or, alternatively, run `pre-commit run --all-files` manually.
