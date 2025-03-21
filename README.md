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

Franky is a high-level control library for Franka robots offering Python and C++ support. 
By wrapping [libfranka](https://frankaemika.github.io/docs/libfranka.html) in a Python interface, Franky eliminates the need for strict real-time programming at 1 kHz. 
Instead, you can define higher-level motion targets in Python, and Franky will use [Ruckig](https://github.com/pantor/ruckig) to plan time-optimal trajectories in real time.

Although Python does not provide real-time guarantees, Franky strives to maintain as much real-time control as possible.
Motions can be preempted at any moment, prompting Franky to replan trajectories on the fly. 
To handle unforeseen situations—such as unexpected contact with the environment—Franky includes a reaction system that allows to dynamically update motion commands.

Check out the [tutorial](#tutorial) and the [examples](https://github.com/TimSchneider42/franky/tree/master/examples) for an introduction.
The full documentation can be found at [https://timschneider42.github.io/franky/](https://timschneider42.github.io/franky/).


## Differences to frankx
Franky is a fork of [frankx](https://github.com/pantor/frankx), though both codebase and functionality differ substantially from frankx by now.
In particular, franky provides the following new features/improvements:
* [Motions can be updated asynchronously.](#real-time-motion)
* [Reactions allow for the registration of callbacks instead of just printing to stdout when fired.](#real-time-reactions)
* [The robot state is also available during control.](#robot-state)
* A larger part of the libfranka API is exposed to python (e.g.,`setCollisionBehavior`, `setJoinImpedance`, and `setCartesianImpedance`).
* Cartesian motion generation handles boundaries in Euler angles properly.
* [There is a new joint motion type that supports waypoints.](#motion-types)
* [The signature of `Affine` changed.](#geometry) `Affine` does not handle elbow positions anymore.
Instead, a new class `RobotPose` stores both the end-effector pose and optionally the elbow position.
* The `MotionData` class does not exist anymore.
Instead, reactions and other settings moved to `Motion`.
* [The `Measure` class allows for arithmetic operations.](#real-time-reactions)
* Exceptions caused by libfranka are raised properly instead of being printed to stdout.
* [We provide wheels for both Franka Research 3 and the older Franka Panda](#installation)

## Setup
To install franky, you have to follow three steps:
1. Ensure that you are using a realtime kernel
2. Ensure that the executing user has permission to run real-time applications
3. Install franky via pip or build it from the sources

### Installing a real-time kernel

In order for franky to function properly, it requires the underlying OS to use a realtime kernel.
Otherwise, you might see `communication_constrains_violation` errors.

To check whether your system is currently using a real-time kernel, type `uname -a`.
You should see something like this:
```
$ uname -a
Linux [PCNAME] 5.15.0-1056-realtime #63-Ubuntu SMP PREEMPT_RT ...
```
If it does not say PREEMPT_RT, you are not currently running a real-time kernel.

There are multiple ways of installing a real-time kernel.
You can [build it from source](https://frankaemika.github.io/docs/installation_linux.html#setting-up-the-real-time-kernel) or, if you are using Ubuntu, it can be [enabled through Ubuntu Pro](https://ubuntu.com/real-time).

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
If realtime is not listed in your groups, try rebooting.

### Installing franky
To start using franky with Python and libfranka *0.15.0*, just install it via
```bash
pip install franky-panda
```
We also provide wheels for libfranka versions *0.5.0*, *0.7.1*, *0.8.0*, *0.9.2*, *0.10.0*, *0.11.0*, *0.12.1*, *0.13.3*, *0.14.2*, and *0.15.0*.
They can be installed via
```bash
VERSION=0-9-2
wget https://github.com/TimSchneider42/franky/releases/latest/download/libfranka_${VERSION}_wheels.zip
unzip libfranka_${VERSION}_wheels.zip
pip install numpy
pip install --no-index --find-links=./dist franky-panda
```

Franky is based on [libfranka](https://github.com/frankaemika/libfranka), [Eigen](https://eigen.tuxfamily.org) for transformation calculations and [pybind11](https://github.com/pybind/pybind11) for the Python bindings.
As the Franka is sensitive to acceleration discontinuities, it requires jerk-constrained motion generation, for which franky uses the [Ruckig](https://ruckig.com) community version for Online Trajectory Generation (OTG).

After installing the dependencies (the exact versions can be found [here](#development)), you can build and install franky via
```bash
git clone --recurse-submodules git@github.com:timschneider42/franky.git
cd franky
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
make install
```

To use franky, you can also include it as a subproject in your parent CMake via `add_subdirectory(franky)` and then `target_link_libraries(<target> franky)`.

If you need only the Python module, you can install franky via
```bash
pip install .
```
Make sure that the built library `_franky.cpython-3**-****-linux-gnu.so` is in the Python path, e.g. by adjusting `PYTHONPATH` accordingly.


#### Using Docker

To use franky within Docker we provide a [Dockerfile](docker/run/Dockerfile) and accompanying [docker-compose](docker-compose.yml) file.

```bash
git clone https://github.com/timschneider42/franky.git
cd franky/
docker compose build franky-run
```

To use another version of libfranka than the default (0.15.0) add a build argument:
```bash
docker compose build franky-run --build-arg LIBFRANKA_VERSION=0.9.2
```

To run the container:
```bash
docker compose run franky-run bash
```
The container requires access to the host machines network *and* elevated user rights to allow the docker user to set RT capabilities of the processes run from within it.


#### Building franky with Docker

For building franky and its wheels, we provide another Docker container that can also be launched using docker-compose:

```bash
docker compose build franky-build
docker compose run --rm franky-build run-tests  # To run the tests
docker compose run --rm franky-build build-wheels  # To build wheels for all supported python versions
```


## Tutorial

Franky comes with both a C++ and Python API that differ only regarding real-time capability.
We will introduce both languages next to each other.
In your C++ project, just include `include <franky.hpp>` and link the library.
For Python, just `import franky`.
As a first example, only four lines of code are needed for simple robotic motions.

```c++
#include <franky.hpp>
using namespace franky;

// Connect to the robot with the FCI IP address
Robot robot("172.16.0.2");

// Reduce velocity and acceleration of the robot
robot.setRelativeDynamicsFactor(0.05);

// Move the end-effector 20cm in positive x-direction
auto motion = std::make_shared<CartesianMotion>(RobotPose(Affine({0.2, 0.0, 0.0}), 0.0), ReferenceType::Relative);

// Finally move the robot
robot.move(motion);
```

The corresponding program in Python is
```python
from franky import Affine, CartesianMotion, Robot, ReferenceType

robot = Robot("172.16.0.2")
robot.relative_dynamics_factor = 0.05

motion = CartesianMotion(Affine([0.2, 0.0, 0.0]), ReferenceType.Relative)
robot.move(motion)
```

Furthermore, we will introduce methods for geometric calculations, for moving the robot according to different motion types, how to implement real-time reactions and changing waypoints in real time as well as controlling the gripper.


### Geometry

`franky.Affine` is a python wrapper for [Eigen::Affine3d](https://eigen.tuxfamily.org/dox/group__TutorialGeometry.html).
It is used for Cartesian poses, frames and transformation.
franky adds its own constructor, which takes a position and a quaternion as inputs:
```python
import math
from scipy.spatial.transform import Rotation
from franky import Affine

z_translation = Affine([0.0, 0.0, 0.5])

quat = Rotation.from_euler("xyz", [0, 0, math.pi / 2]).as_quat()
z_rotation = Affine([0.0, 0.0, 0.0], quat)

combined_transformation = z_translation * z_rotation
```

In all cases, distances are in [m] and rotations in [rad].

### Robot

We wrapped most of the libfanka API (including the RobotState or ErrorMessage) for Python.
Moreover, we added methods to adapt the dynamics of the robot for all motions.
The `rel` name denotes that this a factor of the maximum constraints of the robot.
```python
from franky import Robot

robot = Robot("172.16.0.2")

# Recover from errors
robot.recover_from_errors()

# Set velocity, acceleration and jerk to 5% of the maximum
robot.relative_dynamics_factor = 0.05

# Alternatively, you can define each constraint individually
robot.velocity_rel = 0.2
robot.acceleration_rel = 0.1
robot.jerk_rel = 0.01

# Get the current pose
current_pose = robot.current_pose
```


### Robot State

The robot state can be retrieved by calling the following methods:

* `state`: Object of type `franky.RobotState`, which is a wrapper of the libfranka [franka::RobotState](https://frankaemika.github.io/libfranka/structfranka_1_1RobotState.html) structure.

* `current_cartesian_state`: Object of type `franky.CartesianState`, which contains the end-effector pose and velocity obtained from [franka::RobotState::O_T_EE](https://frankaemika.github.io/libfranka/structfranka_1_1RobotState.html#a193781d47722b32925e0ea7ac415f442) and [franka::RobotState::O_dP_EE_c](https://frankaemika.github.io/libfranka/structfranka_1_1RobotState.html#a4be112bd1a9a7d777a67aea4a18a8dcc).

* `current_joint_position`: Object of type `franky.JointState`, which contains the joint positions and velocities obtained from [franka::RobotState::q](https://frankaemika.github.io/libfranka/structfranka_1_1RobotState.html#ade3335d1ac2f6c44741a916d565f7091) and [franka::RobotState::dq](https://frankaemika.github.io/libfranka/structfranka_1_1RobotState.html#a706045af1b176049e9e56df755325bd2).

```python
robot = Robot("172.16.0.2")

# Get the current state as raw `franky.RobotState`
state = robot.state

# Get the robot's cartesian state
cartesian_state = robot.current_cartesian_state
robot_pose = cartesian_state.pose  # Contains end-effector pose and elbow position
ee_pose = robot_pose.end_effector_pose
elbow_pos = robot_pose.elbow_position
robot_velocity = cartesian_state.velocity  # Contains end-effector twist and elbow velocity
ee_twist = robot_velocity.end_effector_twist
elbow_vel = robot_velocity.elbow_velocity

# Get the robot's joint state
joint_state = robot.current_joint_state
joint_pos = joint_state.position
joint_vel = joint_state.velocity
```


### Motion Types

Franky defines a number of different motion types.
In python, you can use them as follows:
```python
import math
from scipy.spatial.transform import Rotation
from franky import JointWaypointMotion, JointWaypoint, JointPositionStopMotion, CartesianMotion, \
    CartesianWaypointMotion, CartesianWaypoint, Affine, Twist, RobotPose, ReferenceType, CartesianPoseStopMotion, \
    CartesianState, JointState

# A point-to-point motion in the joint space
m1 = JointWaypointMotion([JointWaypoint([-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0.7])])

# A motion in joint space with multiple waypoints
m2 = JointWaypointMotion([
    JointWaypoint([-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0.7]),
    JointWaypoint([0.0, 0.3, 0.3, -1.5, -0.2, 1.5, 0.8]),
    JointWaypoint([0.1, 0.4, 0.3, -1.4, -0.3, 1.7, 0.9])
])

# Intermediate waypoints also permit to specify target velocities. The default target velocity is 0, meaning that the
# robot will stop at every waypoint.
m3 = JointWaypointMotion([
    JointWaypoint([-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0.7]),
    JointWaypoint(
        JointState(
            position=[0.0, 0.3, 0.3, -1.5, -0.2, 1.5, 0.8],
            velocity=[0.1, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0])),
    JointWaypoint([0.1, 0.4, 0.3, -1.4, -0.3, 1.7, 0.9])
])

# Stop the robot
m4 = JointPositionStopMotion()

# A linear motion in cartesian space
quat = Rotation.from_euler("xyz", [0, 0, math.pi / 2]).as_quat()
m5 = CartesianMotion(Affine([0.4, -0.2, 0.3], quat))
m6 = CartesianMotion(RobotPose(Affine([0.4, -0.2, 0.3], quat), elbow_position=0.3))  # With target elbow angle

# A linear motion in cartesian space relative to the initial position
# (Note that this motion is relative both in position and orientation. Hence, when the robot's end-effector is oriented
# differently, it will move in a different direction)
m7 = CartesianMotion(Affine([0.2, 0.0, 0.0]), ReferenceType.Relative)

# Generalization of CartesianMotion that allows for multiple waypoints
m8 = CartesianWaypointMotion([
    CartesianWaypoint(RobotPose(Affine([0.4, -0.2, 0.3], quat), elbow_position=0.3)),
    # The following waypoint is relative to the prior one and 50% slower
    CartesianWaypoint(Affine([0.2, 0.0, 0.0]), ReferenceType.Relative, RelativeDynamicsFactor(0.5, 1.0, 1.0))
])

# Cartesian waypoints also permit to specify target velocities
m9 = CartesianWaypointMotion([
    CartesianWaypoint(Affine([0.5, -0.2, 0.3], quat)),
    CartesianWaypoint(
        CartesianState(
            pose=Affine([0.4, -0.1, 0.3], quat),
            velocity=Twist([-0.01, 0.01, 0.0]))),
    CartesianWaypoint(Affine([0.3, 0.0, 0.3], quat))
])

# Stop the robot. The difference of JointPositionStopMotion to CartesianPoseStopMotion is that JointPositionStopMotion
# stops the robot in joint position control mode while CartesianPoseStopMotion stops it in cartesian pose control mode.
# The difference becomes relevant when asynchronous move commands are being sent (see below).
m10 = CartesianPoseStopMotion()
```

Every motion and waypoint type allows to adapt the dynamics (velocity, acceleration and jerk) by setting the respective `relative_dynamics_factor` parameter.

The real robot can be moved by applying a motion to the robot using `move`:
```python
robot.move(m1)
robot.move(m2)
```


### Real-Time Reactions

By adding reactions to the motion data, the robot can react to unforeseen events.
In the Python API, you can define conditions by using a comparison between a robot's value and a given threshold.
If the threshold is exceeded, the reaction fires.
```python
from franky import CartesianMotion, Affine, ReferenceType, Measure, Reaction

motion = CartesianMotion(Affine([0.0, 0.0, 0.1]), ReferenceType.Relative)  # Move down 10cm

reaction_motion = CartesianMotion(Affine([0.0, 0.0, 0.01]), ReferenceType.Relative)  # Move up for 1cm

# Trigger reaction if the Z force is greater than 30N
reaction = Reaction(Measure.FORCE_Z > 30.0, reaction_motion)
motion.add_reaction(reaction)

robot.move(motion)
```

Possible values to measure are
* `Measure.FORCE_X,` `Measure.FORCE_Y,` `Measure.FORCE_Z`: Force in X, Y and Z direction
* `Measure.REL_TIME`: Time in seconds since the current motion started
* `Measure.ABS_TIME`: Time in seconds since the initial motion started

The difference between `Measure.REL_TIME` and `Measure.ABS_TIME` is that `Measure.REL_TIME` is reset to zero whenever a new motion starts (either by calling `Robot.move` or as a result of a triggered `Reaction`).
`Measure.ABS_TIME`, on the other hand, is only reset to zero when a motion terminates regularly without being interrupted and the robot stops moving.
Hence, `Measure.ABS_TIME` measures the total time in which the robot has moved without interruption.

`Measure` values support all classical arithmetic operations, like addition, subtraction, multiplication, division, and exponentiation (both as base and exponent).
```python
normal_force = (Measure.FORCE_X ** 2 + Measure.FORCE_Y ** 2 + Measure.FORCE_Z ** 2) ** 0.5
```

With arithmetic comparisons, conditions can be generated.
```python
normal_force_within_bounds = normal_force < 30.0
time_up = Measure.ABS_TIME > 10.0
```

Conditions support negation, conjunction (and), and disjunction (or):
```python
abort = ~normal_force_within_bounds | time_up
fast_abort = ~normal_force_within_bounds | time_up
```

To check whether a reaction has fired, a callback can be attached:
```python
from franky import RobotState

def reaction_callback(robot_state: RobotState, rel_time: float, abs_time: float):
    print(f"Reaction fired at {abs_time}.")

reaction.register_callback(reaction_callback)
```

Note that these callbacks are not executed in the control thread since they would otherwise block it.
Instead, they are put in a queue and executed by another thread.
While this scheme ensures that the control thread can always run, it cannot prevent that the queue grows indefinitely when the callbacks take more time to execute than it takes for new callbacks to be queued.
Hence, callbacks might be executed significantly after their respective reaction has fired if they are triggered in rapid succession or take a long time to execute.

In C++ you can additionally use lambdas to define more complex behaviours:
```c++
auto motion = CartesianMotion(RobotPose(Affine({0.0, 0.0, 0.2}), 0.0), ReferenceType::Relative);

// Stop motion if force is over 10N
auto stop_motion = StopMotion<franka::CartesianPose>()

motion
  .addReaction(
    Reaction(
      Measure::ForceZ() > 10.0,  // [N],
      stop_motion))
  .addReaction(
    Reaction(
      Condition(
        [](const franka::RobotState& state, double rel_time, double abs_time) {
          // Lambda condition
          return state.current_errors.self_collision_avoidance_violation;
        }),
      [](const franka::RobotState& state, double rel_time, double abs_time) {
        // Lambda reaction motion generator
        // (we are just returning a stop motion, but there could be arbitrary 
        // logic here for generating reaction motions)
        return StopMotion<franka::CartesianPose>();
      })
    ));

robot.move(motion)
```


### Real-Time Motion

By setting the `asynchronous` parameter of `Robot.move` to `True`, the function does not block until the motion finishes.
Instead, it returns immediately and, thus, allows the main thread to set new motions asynchronously.
```python
import time
from franky import Affine, CartesianMotion, Robot, ReferenceType

robot = Robot("172.16.0.2")
robot.relative_dynamics_factor = 0.05

motion1 = CartesianMotion(Affine([0.2, 0.0, 0.0]), ReferenceType.Relative)
robot.move(motion1, asynchronous=True)

time.sleep(0.5)
motion2 = CartesianMotion(Affine([0.2, 0.0, 0.0]), ReferenceType.Relative)
robot.move(motion2, asynchronous=True)
```

By calling `Robot.join_motion` the main thread can be synchronized with the motion thread, as it will block until the robot finishes its motion.
```python
robot.join_motion()
```

Note that when exceptions occur during the asynchronous execution of a motion, they will not be thrown immediately.
Instead, the control thread stores the exception and terminates.
The next time `Robot.join_motion` or `Robot.move` are called, they will throw the stored exception in the main thread.
Hence, after an asynchronous motion has finished, make sure to call `Robot.join_motion` to ensure being notified of any exceptions that occurred during the motion.


### Gripper

In the `franky::Gripper` class, the default gripper force and gripper speed can be set.
Then, additionally to the libfranka commands, the following helper methods can be used:

```c++
#include <franky.hpp>
#include <chrono>
#include <future>

auto gripper = franky::Gripper("172.16.0.2");

double speed = 0.02; // [m/s]
double force = 20.0; // [N]

// Move the fingers to a specific width (5cm)
bool success = gripper.move(0.05, speed);

// Grasp an object of unknown width
success &= gripper.grasp(0.0, speed, force, epsilon_outer=1.0);

// Get the width of the grasped object
double width = gripper.width();

// Release the object
gripper.open(speed);

// There are also asynchronous versions of the methods
std::future<bool> success_future = gripper.moveAsync(0.05, speed);

// Wait for 1s
if (!success_future.wait_for(std::chrono::seconds(1)) == std::future_status::ready) {
  // Get the result
  std::cout << "Success: " << success_future.get() << std::endl;
} else {
  gripper.stop();
  success_future.wait();
  std::cout << "Gripper motion timed out." << std::endl;
}
```

The Python API follows the c++ API closely:
```python
import franky

gripper = franky.Gripper("172.16.0.2")

speed = 0.02  # [m/s]
force = 20.0  # [N]

# Move the fingers to a specific width (5cm)
success = gripper.move(0.05, speed)

# Grasp an object of unknown width
success &= gripper.grasp(0.0, speed, force, epsilon_outer=1.0)

# Get the width of the grasped object
width = gripper.width

# Release the object
gripper.open(speed)

# There are also asynchronous versions of the methods
success_future = gripper.move_async(0.05, speed)

# Wait for 1s
if success_future.wait(1):
    print(f"Success: {success_future.get()}")
else:
    gripper.stop()
    success_future.wait()
    print("Gripper motion timed out.")
```


## Development

Franky is written in C++17 and Python3.7.
It is currently tested against following versions

- Libfranka 0.5.0, 0.7.1, 0.8.0, 0.9.2, 0.10.0, 0.11.0, 0.12.1, 0.13.3, 0.14.2, 0.15.0
- Eigen 3.4.0
- Pybind11 2.13.6
- Pinocchio 3.4.0
- Python 3.7, 3.8, 3.9, 3.10, 3.11, 3.12, 3.13
- Catch2 2.13.8 (for testing only)

## License

For non-commercial applications, this software is licensed under the LGPL v3.0.
If you want to use franky within commercial applications or under a different license, please contact us for individual agreements.
