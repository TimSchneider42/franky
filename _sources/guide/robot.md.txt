# 🤖 Robot

franky exposes most of the libfanka API for Python.
Moreover, we added methods to adapt the dynamic limits of the robot for all motions.

```python
from franky import *

robot = Robot("10.90.90.1")

# Recover from errors
robot.recover_from_errors()

# Set velocity, acceleration, and jerk to 5% of the maximum
robot.relative_dynamics_factor = 0.05

# Alternatively, you can define each constraint individually
robot.relative_dynamics_factor = RelativeDynamicsFactor(
    velocity=0.1, acceleration=0.05, jerk=0.1
)

# Or, for more fine-grained access, set individual limits
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

## Robot State

The robot state can be retrieved by accessing the following properties:

* `state`: Object of type `franky.RobotState`, which extends the
  libfranka [franka::RobotState](https://frankarobotics.github.io/libfranka/0.15.3/structfranka_1_1RobotState.html) structure by
  additional state elements.
* `current_cartesian_state`: Object of type `franky.CartesianState`, which contains the end-effector pose and velocity
  obtained
  from [franka::RobotState::O_T_EE](https://frankarobotics.github.io/libfranka/0.15.3/structfranka_1_1RobotState.html#a193781d47722b32925e0ea7ac415f442)
  and [franka::RobotState::O_dP_EE_c](https://frankarobotics.github.io/libfranka/0.15.3/structfranka_1_1RobotState.html#a4be112bd1a9a7d777a67aea4a18a8dcc).
* `current_joint_state`: Object of type `franky.JointState`, which contains the joint positions and velocities
  obtained
  from [franka::RobotState::q](https://frankarobotics.github.io/libfranka/0.15.3/structfranka_1_1RobotState.html#ade3335d1ac2f6c44741a916d565f7091)
  and [franka::RobotState::dq](https://frankarobotics.github.io/libfranka/0.15.3/structfranka_1_1RobotState.html#a706045af1b176049e9e56df755325bd2).

```python
from franky import *

robot = Robot("10.90.90.1")

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

# Get the Jacobian of the current robot state
jacobian = robot.model.body_jacobian(Frame.EndEffector, state)

# Alternatively, just get the URDF as a string and do the kinematics computation yourself (only
# for libfranka >= 0.15.0)
urdf_model = robot.model_urdf
```

For a full list of state-related features, check the {py:class}`franky.Robot`
and {py:class}`franky.Model` API documentation.
