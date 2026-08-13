# 🏃‍♂️ Trajectory Control

franky currently supports four trajectory control modes: **joint position control**, **joint velocity control**, **cartesian position control**, and **cartesian velocity control**.
Each of these control modes is invoked by passing the robot an appropriate _Motion_ object.
In addition, franky provides client-side [impedance controllers](impedance-control.md) in torque mode.

In the following, we provide a brief example for each motion type implemented by franky in Python.
The C++ interface is generally analogous, though some variable and method names are different because we
follow [PEP 8](https://peps.python.org/pep-0008/) naming conventions in Python
and [Google naming conventions](https://google.github.io/styleguide/cppguide.html) in C++.

All units are in $m$, $\frac{m}{s}$, $\textit{rad}$, or $\frac{\textit{rad}}{s}$.

For both joint and Cartesian position waypoint motions, a relative waypoint is
resolved against the preceding waypoint's target. The first relative waypoint
is resolved against the commanded position at which the motion starts. This
keeps the path deterministic even when `max_total_duration` advances past a
waypoint before the robot reaches it: subsequent relative waypoints remain
anchored to that waypoint's target, not the position reached at the timeout.

## Joint Position Control

```python
from franky import *

# A point-to-point motion in the joint space
m_jp1 = JointMotion([-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0.7])

# A motion in joint space with multiple waypoints. The robot will stop at each of these
# waypoints. If you want the robot to move continuously, you have to specify a target velocity
# at every waypoint as shown in the example following this one.
m_jp2 = JointWaypointMotion(
    [
        JointWaypoint([-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0.7]),
        JointWaypoint([0.0, 0.3, 0.3, -1.5, -0.2, 1.5, 0.8]),
        JointWaypoint([0.1, 0.4, 0.3, -1.4, -0.3, 1.7, 0.9]),
    ]
)

# Intermediate waypoints also permit specifying target velocities. The default target velocity
# is 0, meaning that the robot will stop at every waypoint.
m_jp3 = JointWaypointMotion(
    [
        JointWaypoint([-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0.7]),
        JointWaypoint(
            JointState(
                position=[0.0, 0.3, 0.3, -1.5, -0.2, 1.5, 0.8],
                velocity=[0.1, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0],
            )
        ),
        JointWaypoint([0.1, 0.4, 0.3, -1.4, -0.3, 1.7, 0.9]),
    ]
)

# Stop the robot in joint position control mode. The difference between JointStopMotion to other
# stop-motions, such as CartesianStopMotion, is that JointStopMotion stops the robot in joint
# position control mode while CartesianStopMotion stops it in cartesian pose control mode. The
# difference becomes relevant when asynchronous move commands are being sent or reactions are
# being used(see below).
m_jp4 = JointStopMotion()
```

## Joint Velocity Control

```python
from franky import *

# Accelerate to the given joint velocity and hold it. After 1000ms, stop the robot again.
m_jv1 = JointVelocityMotion(
    [0.1, 0.3, -0.1, 0.0, 0.1, -0.2, 0.4], duration=Duration(1000)
)

# Joint velocity motions also support waypoints. Unlike in joint position control, a joint
# velocity waypoint is a target velocity to be reached. This particular example first
# accelerates the joints, holds the velocity for 1s, then reverses direction for 2s, reverses
# direction again for 1s, and finally stops. It is important not to forget to stop the robot
# at the end of such a sequence, as it will otherwise throw an error.
m_jv2 = JointVelocityWaypointMotion(
    [
        JointVelocityWaypoint(
            [0.1, 0.3, -0.1, 0.0, 0.1, -0.2, 0.4], hold_target_duration=Duration(1000)
        ),
        JointVelocityWaypoint(
            [-0.1, -0.3, 0.1, -0.0, -0.1, 0.2, -0.4],
            hold_target_duration=Duration(2000),
        ),
        JointVelocityWaypoint(
            [0.1, 0.3, -0.1, 0.0, 0.1, -0.2, 0.4], hold_target_duration=Duration(1000)
        ),
        JointVelocityWaypoint([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]
)

# Stop the robot in joint velocity control mode.
m_jv3 = JointVelocityStopMotion()
```

## Cartesian Position Control

```python
import math
from scipy.spatial.transform import Rotation
from franky import *

# Move to the given target pose
quat = Rotation.from_euler("xyz", [0, 0, math.pi / 2]).as_quat()
m_cp1 = CartesianMotion(Affine([0.4, -0.2, 0.3], quat))

# With target elbow angle (otherwise, the Franka firmware will choose by itself)
m_cp2 = CartesianMotion(
    RobotPose(Affine([0.4, -0.2, 0.3], quat), elbow_state=ElbowState(0.3))
)

# A linear motion in Cartesian space relative to the initial position
# (Note that this motion is relative both in position and orientation. Hence, when the robot's
# end-effector is oriented differently, it will move in a different direction)
m_cp3 = CartesianMotion(Affine([0.2, 0.0, 0.0]), ReferenceType.Relative)

# Generalization of CartesianMotion that allows for multiple waypoints. The robot will stop at
# each of these waypoints. If you want the robot to move continuously, you have to specify a
# target velocity at every waypoint as shown in the example following this one.
m_cp4 = CartesianWaypointMotion(
    [
        CartesianWaypoint(
            RobotPose(Affine([0.4, -0.2, 0.3], quat), elbow_state=ElbowState(0.3))
        ),
        # The following waypoint is relative to the prior one and 50% slower
        CartesianWaypoint(
            Affine([0.2, 0.0, 0.0]),
            ReferenceType.Relative,
            RelativeDynamicsFactor(0.5, 1.0, 1.0),
        ),
    ]
)

# Cartesian waypoints permit specifying target velocities
m_cp5 = CartesianWaypointMotion(
    [
        CartesianWaypoint(Affine([0.5, -0.2, 0.3], quat)),
        CartesianWaypoint(
            CartesianState(
                pose=Affine([0.4, -0.1, 0.3], quat), velocity=Twist([-0.01, 0.01, 0.0])
            )
        ),
        CartesianWaypoint(Affine([0.3, 0.0, 0.3], quat)),
    ]
)

# Stop the robot in Cartesian position control mode.
m_cp6 = CartesianStopMotion()
```

Cartesian poses specify a final orientation, not a rotation direction or
number of turns. Each waypoint therefore uses an equivalent rotation of at
most 180°. To request a larger rotation in a particular direction, split it
into consistently signed relative increments smaller than 180° (for example,
two +135° waypoints for +270°).

## Cartesian Velocity Control

```python
from franky import *

# A Cartesian velocity motion with linear (first argument) and angular (second argument)
# components
m_cv1 = CartesianVelocityMotion(Twist([0.2, -0.1, 0.1], [0.1, -0.1, 0.2]))

# With target elbow velocity
m_cv2 = CartesianVelocityMotion(
    RobotVelocity(Twist([0.2, -0.1, 0.1], [0.1, -0.1, 0.2]), elbow_velocity=-0.2)
)

# Cartesian velocity motions also support multiple waypoints. Unlike in Cartesian position
# control, a Cartesian velocity waypoint is a target velocity to be reached. This particular
# example first accelerates the end-effector, holds the velocity for 1s, then reverses
# direction for 2s, reverses direction again for 1s, and finally stops. It is important not to
# forget to stop the robot at the end of such a sequence, as it will otherwise throw an error.
m_cv4 = CartesianVelocityWaypointMotion(
    [
        CartesianVelocityWaypoint(
            Twist([0.2, -0.1, 0.1], [0.1, -0.1, 0.2]),
            hold_target_duration=Duration(1000),
        ),
        CartesianVelocityWaypoint(
            Twist([-0.2, 0.1, -0.1], [-0.1, 0.1, -0.2]),
            hold_target_duration=Duration(2000),
        ),
        CartesianVelocityWaypoint(
            Twist([0.2, -0.1, 0.1], [0.1, -0.1, 0.2]),
            hold_target_duration=Duration(1000),
        ),
        CartesianVelocityWaypoint(Twist()),
    ]
)

# Stop the robot in Cartesian velocity control mode.
m_cv6 = CartesianVelocityStopMotion()
```
