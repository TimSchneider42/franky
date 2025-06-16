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
By providing a high-level control interface, Franky eliminates the need for strict real-time programming at 1 kHz,
making control from non-real-time environments, such as Python programs, feasible.
Instead of relying on low-level control commands, Franky expects high-level position or velocity targets and
uses [Ruckig](https://github.com/pantor/ruckig) to plan time-optimal trajectories in real-time.

Although Python does not provide real-time guarantees, Franky strives to maintain as much real-time control as possible.
Motions can be preempted at any moment, prompting Franky to re-plan trajectories on the fly.
To handle unforeseen situations—such as unexpected contact with the environment — Franky includes a reaction system that
allows to update motion commands dynamically.
Furthermore, most non-real-time functionality of [libfranka](https://frankaemika.github.io/docs/libfranka.html), such as
Gripper control is made directly available in Python.

Check out the [tutorial](#-tutorial) and the [examples](https://github.com/TimSchneider42/franky/tree/master/examples)
for an introduction.
The full documentation can be found
at [https://timschneider42.github.io/franky/](https://timschneider42.github.io/franky/).

## 🚀 Features

- **Control your Franka robot directly from Python in just a few lines!**
  No more endless hours setting up ROS, juggling packages, or untangling dependencies. Just `pip install` — no ROS at all.

- **[Four control modes](#motion-types)**: [Cartesian position](#cartesian-position-control), [Cartesian velocity](#cartesian-velocity-control), [Joint position](#joint-position-control), [Joint velocity](#joint-velocity-control)
  Franky uses [Ruckig](https://github.com/pantor/ruckig) to generate smooth, time-optimal trajectories while respecting velocity, acceleration, and jerk limits.

- **[Real-time control from Python and C++](#real-time-motions)**
  Need to change the target while the robot’s moving? No problem. Franky re-plans trajectories on the fly so that you can preempt motions anytime.

- **[Reactive behavior](#-real-time-reactions)**
  Robots don’t always go according to plan. Franky lets you define reactions to unexpected events—like contact with the environment — so you can change course in real-time.

- **[Motion and reaction callbacks](#motion-callbacks)**
  Want to monitor what’s happening under the hood? Add callbacks to your motions and reactions. They won’t block the control thread and are super handy for debugging or logging.

- **Things are moving too fast? [Tune the robot's dynamics to your needs](#-robot)**
  Adjust max velocity, acceleration, and jerk to match your setup or task. Fine control for smooth, safe operation.

- **Full Python access to the libfranka API**
  Want to tweak impedance, read the robot state, set force thresholds, or mess with the Jacobian? Go for it. If libfranka supports it, chances are Franky does, too.

## 📖 Python Quickstart Guide

Real-time kernel already installed and real-time permissions granted? Just install Franky via

```bash
pip install franky-control
```

otherwise, follow the [setup instructions](#setup) first.

Now we are already ready to go!
Unlock the brakes in the web interface, activate FCI, and start coding:
```python
from franky import *

robot = Robot("172.16.0.2")  # Replace this with your robot's IP

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
then your Franka robot is either not on the most recent firmware version or you are using the older Franka Panda model.
In any case, it's no big deal; just check [here](https://frankaemika.github.io/docs/compatibility.html) which libfranka version you need and follow our [instructions](installing-frankly) to install the appropriate Franky wheels.

## <a id="setup" /> ⚙️ Setup

To install Franky, you have to follow three steps:

1. Ensure that you are using a realtime kernel
2. Ensure that the executing user has permission to run real-time applications
3. Install Franky via pip or build it from source

### Installing a real-time kernel

In order for Franky to function properly, it requires the underlying OS to use a realtime kernel.
Otherwise, you might see `communication_constrains_violation` errors.

To check whether your system is currently using a real-time kernel, type `uname -a`.
You should see something like this:

```
$ uname -a
Linux [PCNAME] 5.15.0-1056-realtime #63-Ubuntu SMP PREEMPT_RT ...
```

If it does not say PREEMPT_RT, you are not currently running a real-time kernel.

There are multiple ways of installing a real-time kernel.
You
can [build it from source](https://frankaemika.github.io/docs/installation_linux.html#setting-up-the-real-time-kernel)
or, if you are using Ubuntu, it can be [enabled through Ubuntu Pro](https://ubuntu.com/real-time).

### Allowing the executing user to run real-time applications

First, create a group `realtime` and add your user (or whoever is running Franky) to this group:

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

### Installing Franky

To start using Franky with Python and libfranka *0.15.0*, just install it via

```bash
pip install franky-control
```

We also provide wheels for libfranka versions *0.7.1*, *0.8.0*, *0.9.2*, *0.10.0*, *0.11.0*, *0.12.1*, *0.13.3*,
*0.14.2*, and *0.15.0*.
They can be installed via

```bash
VERSION=0-9-2
wget https://github.com/TimSchneider42/franky/releases/latest/download/libfranka_${VERSION}_wheels.zip
unzip libfranka_${VERSION}_wheels.zip
pip install numpy
pip install --no-index --find-links=./dist franky-control
```

Franky is based on [libfranka](https://github.com/frankaemika/libfranka), [Eigen](https://eigen.tuxfamily.org) for
transformation calculations and [pybind11](https://github.com/pybind/pybind11) for the Python bindings.
As the Franka is sensitive to acceleration discontinuities, it requires jerk-constrained motion generation, for which
Franky uses the [Ruckig](https://ruckig.com) community version for Online Trajectory Generation (OTG).

After installing the dependencies (the exact versions can be found [here](#-development)), you can build and install
Franky via

```bash
git clone --recurse-submodules git@github.com:timschneider42/franky.git
cd franky
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
make install
```

To use Franky, you can also include it as a subproject in your parent CMake via `add_subdirectory(franky)` and then
`target_link_libraries(<target> franky)`.

If you need only the Python module, you can install Franky via

```bash
pip install .
```

Make sure that the built library `_franky.cpython-3**-****-linux-gnu.so` is in the Python path, e.g. by adjusting
`PYTHONPATH` accordingly.

#### Using Docker

To use Franky within Docker we provide a [Dockerfile](docker/run/Dockerfile) and
accompanying [docker-compose](docker-compose.yml) file.

```bash
git clone --recurse-submodules https://github.com/timschneider42/franky.git
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

The container requires access to the host machines network *and* elevated user rights to allow the docker user to set RT
capabilities of the processes run from within it.

#### Building Franky with Docker

For building Franky and its wheels, we provide another Docker container that can also be launched using docker-compose:

```bash
docker compose build franky-build
docker compose run --rm franky-build run-tests  # To run the tests
docker compose run --rm franky-build build-wheels  # To build wheels for all supported python versions
```

### Can I use CUDA jointly with Franky?

Yes. However, you need to set `IGNORE_PREEMPT_RT_PRESENCE=1` during the installation and all subsequent updates of the CUDA drivers on the real-time kernel.

First, make sure that you have rebooted your system after installing the real-time kernel.
Then, add `IGNORE_PREEMPT_RT_PRESENCE=1` to `/etc/environment`, call `export IGNORE_PREEMPT_RT_PRESENCE=1` to also set it in the current session and follow the instructions of Nvidia to install CUDA on your system.

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

## 📚 Tutorial

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

Before executing any code, make sure that you have enabled the Franka Control Interface (FCI) in the Franka UI web interface.

Furthermore, we will introduce methods for geometric calculations, for moving the robot according to different motion
types, how to implement real-time reactions and changing waypoints in real time as well as controlling the gripper.

### 🧮 Geometry

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

### 🤖 Robot

Franky exposes most of the libfanka API for Python.
Moreover, we added methods to adapt the dynamics limits of the robot for all motions.

```python
from franky import *

robot = Robot("172.16.0.2")

# Recover from errors
robot.recover_from_errors()

# Set velocity, acceleration and jerk to 5% of the maximum
robot.relative_dynamics_factor = 0.05

# Alternatively, you can define each constraint individually
robot.relative_dynamics_factor = RelativeDynamicsFactor(velocity=0.1, acceleration=0.05, jerk=0.1)

# Or, for more finegrained access, set individual limits
robot.translation_velocity_limit.set(3.0)
robot.rotation_velocity_limit.set(2.5)
robot.elbow_velocity_limit.set(2.62)
robot.translation_acceleration_limit.set(9.0)
robot.rotation_acceleration_limit.set(17.0)
robot.elbow_acceleration_limit.set(10.0)
robot.translation_jerk_limit.set(4500.0)
robot.rotation_jerk_limit.set(8500.0)
robot.elbow_jerk_limit.set(5000.0)
robot.joint_velocity_limit.set([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
robot.joint_acceleration_limit.set([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
robot.joint_jerk_limit.set([5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0])
# By default, these limits are set to their respective maxima (the values shown here)

# Get the max of each limit (as provided by Franka) with the max function, e.g.:
print(robot.joint_jerk_limit.max)
```

#### Robot State

The robot state can be retrieved by accessing the following properties:

* `state`: Object of type `franky.RobotState`, which extends the
  libfranka [franka::RobotState](https://frankaemika.github.io/libfranka/structfranka_1_1RobotState.html) structure by
  additional state elements.
* `current_cartesian_state`: Object of type `franky.CartesianState`, which contains the end-effector pose and velocity
  obtained
  from [franka::RobotState::O_T_EE](https://frankaemika.github.io/libfranka/structfranka_1_1RobotState.html#a193781d47722b32925e0ea7ac415f442)
  and [franka::RobotState::O_dP_EE_c](https://frankaemika.github.io/libfranka/structfranka_1_1RobotState.html#a4be112bd1a9a7d777a67aea4a18a8dcc).
* `current_joint_state`: Object of type `franky.JointState`, which contains the joint positions and velocities
  obtained
  from [franka::RobotState::q](https://frankaemika.github.io/libfranka/structfranka_1_1RobotState.html#ade3335d1ac2f6c44741a916d565f7091)
  and [franka::RobotState::dq](https://frankaemika.github.io/libfranka/structfranka_1_1RobotState.html#a706045af1b176049e9e56df755325bd2).

```python
from franky import *

robot = Robot("172.16.0.2")

# Get the current state as `franky.RobotState`. See the documentation for a list of fields.
state = robot.state

# Get the robot's cartesian state
cartesian_state = robot.current_cartesian_state
robot_pose = cartesian_state.pose  # Contains end-effector pose and elbow position
ee_pose = robot_pose.end_effector_pose
elbow_pos = robot_pose.elbow_state
robot_velocity = cartesian_state.velocity  # Contains end-effector twist and elbow velocity
ee_twist = robot_velocity.end_effector_twist
elbow_vel = robot_velocity.elbow_velocity

# Get the robot's joint state
joint_state = robot.current_joint_state
joint_pos = joint_state.position
joint_vel = joint_state.velocity

# Use the robot model to compute kinematics
q = [-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0.7]
f_t_ee = Affine()
ee_t_k = Affine()
ee_pose_kin = robot.model.pose(Frame.EndEffector, q, f_t_ee, ee_t_k)

# Get the jacobian of the current robot state
jacobian = robot.model.body_jacobian(Frame.EndEffector, state)

# Alternatively, just get the URDF as string and do the kinematics computation yourself (only for libfranka >= 0.15.0)
urdf_model = robot.model_urdf
```

For a full list of state-related features, check
the [Robot](https://timschneider42.github.io/franky/classfranky_1_1_robot.html)
and [Model](https://timschneider42.github.io/franky/classfranky_1_1_model.html) sections of the documentation.

### <a id="motion-types" /> 🏃‍♂️ Motion Types

Franky currently supports four different impedance control modes: **joint position control**, **joint velocity control**, **cartesian position control**, and **cartesian velocity control**.
Each of these control modes is invoked by passing the robot an appropriate _Motion_ object.

In the following, we provide a brief example for each motion type implemented by Franky in Python.
The C++ interface is generally analogous, though some variable and method names are different because we
follow [PEP 8](https://peps.python.org/pep-0008/) naming conventions in Python
and [Google naming conventions](https://google.github.io/styleguide/cppguide.html) in C++.

All units are in $m$, $\frac{m}{s}$, $\textit{rad}$, or $\frac{\textit{rad}}{s}$.

#### Joint Position Control

```python
from franky import *

# A point-to-point motion in the joint space
m_jp1 = JointMotion([-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0.7])

# A motion in joint space with multiple waypoints
m_jp2 = JointWaypointMotion([
    JointWaypoint([-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0.7]),
    JointWaypoint([0.0, 0.3, 0.3, -1.5, -0.2, 1.5, 0.8]),
    JointWaypoint([0.1, 0.4, 0.3, -1.4, -0.3, 1.7, 0.9])
])

# Intermediate waypoints also permit to specify target velocities. The default target velocity is 0, meaning that the
# robot will stop at every waypoint.
m_jp3 = JointWaypointMotion([
    JointWaypoint([-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0.7]),
    JointWaypoint(
        JointState(
            position=[0.0, 0.3, 0.3, -1.5, -0.2, 1.5, 0.8],
            velocity=[0.1, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0])),
    JointWaypoint([0.1, 0.4, 0.3, -1.4, -0.3, 1.7, 0.9])
])

# Stop the robot in joint position control mode. The difference of JointStopMotion to other stop motions such as
# CartesianStopMotion is that # JointStopMotion # stops the robot in joint position control mode while
# CartesianStopMotion stops it in cartesian pose control mode. The difference becomes relevant when asynchronous move
# commands are being sent or reactions are being used(see below).
m_jp4 = JointStopMotion()
```

#### Joint Velocity Control

```python
from franky import *

# Accelerate to the given joint velocity and hold it. After 1000ms stop the robot again.
m_jv1 = JointVelocityMotion([0.1, 0.3, -0.1, 0.0, 0.1, -0.2, 0.4], duration=Duration(1000))

# Joint velocity motions also support waypoints. Unlike in joint position control, a joint velocity waypoint is a
# target velocity to be reached. This particular example first accelerates the joints, holds the velocity for 1s, then
# reverses direction for 2s, reverses direction again for 1s, and finally stops. It is important not to forget to stop
# the robot at the end of such a sequence, as it will otherwise throw an error.
m_jv2 = JointVelocityWaypointMotion([
    JointVelocityWaypoint([0.1, 0.3, -0.1, 0.0, 0.1, -0.2, 0.4], hold_target_duration=Duration(1000)),
    JointVelocityWaypoint([-0.1, -0.3, 0.1, -0.0, -0.1, 0.2, -0.4], hold_target_duration=Duration(2000)),
    JointVelocityWaypoint([0.1, 0.3, -0.1, 0.0, 0.1, -0.2, 0.4], hold_target_duration=Duration(1000)),
    JointVelocityWaypoint([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
])

# Stop the robot in joint velocity control mode.
m_jv3 = JointVelocityStopMotion()
```

#### Cartesian Position Control

```python
import math
from scipy.spatial.transform import Rotation
from franky import *

# Move to the given target pose
quat = Rotation.from_euler("xyz", [0, 0, math.pi / 2]).as_quat()
m_cp1 = CartesianMotion(Affine([0.4, -0.2, 0.3], quat))

# With target elbow angle (otherwise, the Franka firmware will choose by itself)
m_cp2 = CartesianMotion(RobotPose(Affine([0.4, -0.2, 0.3], quat), elbow_state=ElbowState(0.3)))

# A linear motion in cartesian space relative to the initial position
# (Note that this motion is relative both in position and orientation. Hence, when the robot's end-effector is oriented
# differently, it will move in a different direction)
m_cp3 = CartesianMotion(Affine([0.2, 0.0, 0.0]), ReferenceType.Relative)

# Generalization of CartesianMotion that allows for multiple waypoints
m_cp4 = CartesianWaypointMotion([
    CartesianWaypoint(RobotPose(Affine([0.4, -0.2, 0.3], quat), elbow_state=ElbowState(0.3))),
    # The following waypoint is relative to the prior one and 50% slower
    CartesianWaypoint(Affine([0.2, 0.0, 0.0]), ReferenceType.Relative, RelativeDynamicsFactor(0.5, 1.0, 1.0))
])

# Cartesian waypoints also permit to specify target velocities
m_cp5 = CartesianWaypointMotion([
    CartesianWaypoint(Affine([0.5, -0.2, 0.3], quat)),
    CartesianWaypoint(
        CartesianState(
            pose=Affine([0.4, -0.1, 0.3], quat),
            velocity=Twist([-0.01, 0.01, 0.0]))),
    CartesianWaypoint(Affine([0.3, 0.0, 0.3], quat))
])

# Stop the robot in cartesian position control mode.
m_cp6 = CartesianStopMotion()
```

#### Cartesian Velocity Control

```python
from franky import *

# A cartesian velocity motion with linear (first argument) and angular (second argument) components
m_cv1 = CartesianVelocityMotion(Twist([0.2, -0.1, 0.1], [0.1, -0.1, 0.2]))

# With target elbow velocity
m_cv2 = CartesianVelocityMotion(RobotVelocity(Twist([0.2, -0.1, 0.1], [0.1, -0.1, 0.2]), elbow_velocity=-0.2))

# Cartesian velocity motions also support multiple waypoints. Unlike in cartesian position control, a cartesian velocity
# waypoint is a target velocity to be reached. This particular example first accelerates the end-effector, holds the
# velocity for 1s, then # reverses direction for 2s, reverses direction again for 1s, and finally stops. It is important
# not to forget to stop # the robot at the end of such a sequence, as it will otherwise throw an error.
m_cv4 = CartesianVelocityWaypointMotion([
    CartesianVelocityWaypoint(Twist([0.2, -0.1, 0.1], [0.1, -0.1, 0.2]), hold_target_duration=Duration(1000)),
    CartesianVelocityWaypoint(Twist([-0.2, 0.1, -0.1], [-0.1, 0.1, -0.2]), hold_target_duration=Duration(2000)),
    CartesianVelocityWaypoint(Twist([0.2, -0.1, 0.1], [0.1, -0.1, 0.2]), hold_target_duration=Duration(1000)),
    CartesianVelocityWaypoint(Twist()),
])

# Stop the robot in cartesian velocity control mode.
m_cv6 = CartesianVelocityStopMotion()
```

#### Relative Dynamics Factors

Every motion and waypoint type allows to adapt the dynamics (velocity, acceleration and jerk) by setting the respective
`relative_dynamics_factor` parameter.
This parameter can also be set for the robot globally as shown below or in the `robot.move` command.
Crucially, relative dynamics factors on different layers (robot, move command, and motion) do not override each other
but rather get multiplied.
Hence, a relative dynamics factor on a motion can only reduce the dynamics of the robot and never increase them.

There is one exception to this rule and that is if any layer sets the relative dynamics factor to
`RelativeDynamicsFactor.MAX_DYNAMICS`.
This will cause the motion to be executed with maximum velocity, acceleration, and jerk limits, independently of the
relative dynamics factors of the other layers.
This feature should only be used to abruptly stop the robot in case of an unexpected environment contact as executing
other motions with it is likely to lead to a discontinuity error and might be dangerous.

#### Executing Motions

The real robot can be moved by applying a motion to the robot using `move`:

```python
# Before moving the robot, set an appropriate dynamics factor. We start small:
robot.relative_dynamics_factor = 0.05
# or alternatively, to control the scaling of velocity, acceleration, and jerk limits separately:
robot.relative_dynamics_factor = RelativeDynamicsFactor(0.05, 0.1, 0.15)
# If these values are set too high, you will see discontinuity errors

robot.move(m_jp1)

# We can also set a relative dynamics factor in the move command. It will be multiplied with the other relative
# dynamics factors (robot and motion if present).
robot.move(m_jp2, relative_dynamics_factor=0.8)
```

#### Motion Callbacks

All motions support callbacks, which will be invoked in every control step at 1kHz.
Callbacks can be attached as follows:

```python
def cb(
        robot_state: RobotState,
        time_step: Duration,
        rel_time: Duration,
        abs_time: Duration,
        control_signal: JointPositions):
    print(f"At time {abs_time}, the target joint positions were {control_signal.q}")


m_jp1.register_callback(cb)
robot.move(m_jp1)
```

Note that in Python, these callbacks are not executed in the control thread since they would otherwise block it.
Instead, they are put in a queue and executed by another thread.
While this scheme ensures that the control thread can always run, it cannot prevent that the queue grows indefinitely
when the callbacks take more time to execute than it takes for new callbacks to be queued.
Hence, callbacks might be executed significantly after they were queued if they take a long time to execute.

### ⚡ Real-Time Reactions

By adding reactions to the motion data, the robot can react to unforeseen events.
In the Python API, you can define conditions by using a comparison between a robot's value and a given threshold.
If the threshold is exceeded, the reaction fires.

```python
from franky import CartesianMotion, Affine, ReferenceType, Measure, Reaction

motion = CartesianMotion(Affine([0.0, 0.0, 0.1]), ReferenceType.Relative)  # Move down 10cm

# It is important that the reaction motion uses the same control mode as the original motion. Hence, we cannot register
# a JointMotion as a reaction motion to a CartesianMotion.
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

The difference between `Measure.REL_TIME` and `Measure.ABS_TIME` is that `Measure.REL_TIME` is reset to zero whenever a
new motion starts (either by calling `Robot.move` or as a result of a triggered `Reaction`).
`Measure.ABS_TIME`, on the other hand, is only reset to zero when a motion terminates regularly without being
interrupted and the robot stops moving.
Hence, `Measure.ABS_TIME` measures the total time in which the robot has moved without interruption.

`Measure` values support all classical arithmetic operations, like addition, subtraction, multiplication, division, and
exponentiation (both as base and exponent).

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

Similar to the motion callbacks, in Python, reaction callbacks are not executed in real-time but in a regular thread
with lower priority to ensure that the control thread does not get blocked.
Thus, the callbacks might fire substantially after the reaction has fired, depending on the time it takes to execute
them.

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

###  <a id="real-time-motions" /> ⏱️ Real-Time Motions

By setting the `asynchronous` parameter of `Robot.move` to `True`, the function does not block until the motion
finishes.
Instead, it returns immediately and, thus, allows the main thread to set new motions asynchronously.

```python
import time
from franky import Affine, CartesianMotion, Robot, ReferenceType

robot = Robot("172.16.0.2")
robot.relative_dynamics_factor = 0.05

motion1 = CartesianMotion(Affine([0.2, 0.0, 0.0]), ReferenceType.Relative)
robot.move(motion1, asynchronous=True)

time.sleep(0.5)
# Note that similar to reactions, when preempting active motions with new motions, the control mode cannot change.
# Hence, we cannot use, e.g., a JointMotion here.
motion2 = CartesianMotion(Affine([0.2, 0.0, 0.0]), ReferenceType.Relative)
robot.move(motion2, asynchronous=True)
```

By calling `Robot.join_motion` the main thread can be synchronized with the motion thread, as it will block until the
robot finishes its motion.

```python
robot.join_motion()
```

Note that when exceptions occur during the asynchronous execution of a motion, they will not be thrown immediately.
Instead, the control thread stores the exception and terminates.
The next time `Robot.join_motion` or `Robot.move` are called, they will throw the stored exception in the main thread.
Hence, after an asynchronous motion has finished, make sure to call `Robot.join_motion` to ensure being notified of any
exceptions that occurred during the motion.

### <a id="gripper" /> 👌  Gripper

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

### Accessing the Web Interface API

For Franka robots, control happens via the Franka Control Interface (FCI), which has to be enabled through the Franka UI in the robot's web interface.
The Franka UI also provides methods for locking and unlocking the brakes, setting the execution mode, and executing the safety self-test.
However, sometimes you may want to access these methods programmatically, e.g. for automatically unlocking the brakes before starting a motion, or automatically executing the self-test after 24h of continuous execution.

For that reason, Franky provides a `RobotWebSession` class that allows you to access the web interface API of the robot.
Note that directly accessing the web interface API is not officially supported and documented by Franka.
Hence, use this feature at your own risk.

A typical automated workflow could look like this:

```python
import franky

with franky.RobotWebSession("172.16.0.2", "username", "password") as robot_web_session:
    # First take control, in case some other web session is currently running
    assert robot_web_session.take_control(), "Control not granted"

    # Unlock the brakes
    robot_web_session.unlock_brakes()

    # Enable the FCI
    robot_web_session.enable_fci()

    # Create a franky.Robot instance and do whatever you want
    ...

    # Disable the FCI
    robot_web_session.disable_fci()

    # Lock brakes
    robot_web_session.lock_brakes()
```

In case you are running the robot for longer than 24h you will have noticed that you have to do a safety self-test every 24h.
`RobotWebSession` allows to automate this task as well:

```python
import time
import franky

with franky.RobotWebSession("172.16.0.2", "username", "password") as robot_web_session:
    # Execute self-test if the time until self-test is less than 5 minutes.
    if robot_web_session.get_system_status()["safety"]["timeToTd2"] < 300:
        robot_web_session.disable_fci()
        robot_web_session.lock_brakes()
        time.sleep(1.0)

        robot_web_session.execute_self_test()

        robot_web_session.unlock_brakes()
        robot_web_session.enable_fci()
        time.sleep(1.0)

        # Recreate your franky.Robot instance as the FCI has been disabled and re-enabled
        ...
```

`robot_web_session.get_system_status()` contains more information than just the time until self-test, such as the current execution mode, whether the brakes are locked, whether the FCI is enabled, and more.

If you want to call other API functions, you can use the `RobotWebSession.send_api_request` and `RobotWebSession.send_control_api_request` methods.
See [robot_web_session.py](franky/robot_web_session.py) for an example of how to use these methods.

## 🛠️ Development

Franky is currently tested against following versions

- libfranka 0.7.1, 0.8.0, 0.9.2, 0.10.0, 0.11.0, 0.12.1, 0.13.3, 0.14.2, 0.15.0
- Eigen 3.4.0
- Pybind11 2.13.6
- POCO 1.12.5p2
- Pinocchio 3.4.0
- Python 3.7, 3.8, 3.9, 3.10, 3.11, 3.12, 3.13
- Catch2 2.13.8 (for testing only)

## 📜 License

For non-commercial applications, this software is licensed under the LGPL v3.0.
If you want to use Franky within commercial applications or under a different license, please contact us for individual
agreements.

## 🔍 Differences to frankx

Franky started originally as a fork of [frankx](https://github.com/pantor/frankx), though both codebase and
functionality differ substantially from frankx by now.
Aside of bug fixes and general performance improvements, Franky provides the following new features/improvements:

* [Motions can be updated asynchronously.](#-real-time-motions)
* [Reactions allow for the registration of callbacks instead of just printing to stdout when fired.](#-real-time-reactions)
* [Motions allow for the registration of callbacks for profiling.](#motion-callbacks)
* [The robot state is also available during control.](#robot-state)
* A larger part of the libfranka API is exposed to python (e.g.,`setCollisionBehavior`, `setJoinImpedance`, and
  `setCartesianImpedance`).
* Cartesian motion generation handles boundaries in Euler angles properly.
* [There is a new joint motion type that supports waypoints.](#-motion-types)
* [The signature of `Affine` changed.](#-geometry) `Affine` does not handle elbow positions anymore.
  Instead, a new class `RobotPose` stores both the end-effector pose and optionally the elbow position.
* The `MotionData` class does not exist anymore.
  Instead, reactions and other settings moved to `Motion`.
* [The `Measure` class allows for arithmetic operations.](#-real-time-reactions)
* Exceptions caused by libfranka are raised properly instead of being printed to stdout.
* [We provide wheels for both Franka Research 3 and the older Franka Panda](#-setup)
* Franky supports [joint velocity control](#joint-velocity-control)
  and [cartesian velocity control](#cartesian-velocity-control)
* The dynamics limits are not hard-coded anymore but can be [set for each robot instance](#-robot).

## Contributing

If you wish to contribute to this project, you are welcome to create a pull request.
Please run the [pre-commit](https://pre-commit.com/) hooks before submitting your pull request.
To install the pre-commit hooks, run:

1. [Install pre-commit](https://pre-commit.com/#install)
2. Install the Git hooks by running `pre-commit install` or, alternatively, run `pre-commit run --all-files manually.
