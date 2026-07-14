# 📚 Guide

franky comes with both a C++ and Python API that differ only regarding real-time capability.
We will introduce both languages next to each other.
In your C++ project, just include `include <franky.hpp>` and link the library.
For Python, just `import franky`.
As a first example, only four lines of code are needed for simple robotic motions.

```c++
#include <franky.hpp>
using namespace franky;

// Connect to the robot with the FCI IP address
Robot robot("10.90.90.1");

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

robot = Robot("10.90.90.1")
robot.relative_dynamics_factor = 0.05

motion = CartesianMotion(Affine([0.2, 0.0, 0.0]), ReferenceType.Relative)
robot.move(motion)
```

Before executing any code, make sure that you have enabled the Franka Control Interface (FCI) in the Franka UI web
interface.

The following chapters introduce methods for geometric calculations, for moving the robot according to different motion
types, how to implement real-time reactions and changing waypoints in real time, as well as controlling the gripper.

```{toctree}
:maxdepth: 2

geometry
robot
motion-types
impedance-control
executing-motions
reactions
realtime-motions
gripper
desk
simulation
```
