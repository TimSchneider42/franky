# 🐍 Python API

All classes listed here are available directly from the top-level `franky` module, e.g. `from franky import Robot`.
For a task-oriented introduction, see the [guide](../guide/index.md).

## Robot

The central classes for connecting to and configuring the robot.
See the [Robot](../guide/robot.md) chapter of the guide for an introduction.

```{eval-rst}
.. currentmodule:: franky

.. autosummary::
   :toctree: generated
   :nosignatures:

   Robot
   Model
   Frame
   RobotState
   Errors
   RobotMode
   RealtimeConfig
   ControllerMode
   ControlSignalType
   Duration
   RelativeDynamicsFactor
   DoubleDynamicsLimit
   VectorDynamicsLimit
```

## Geometry and Kinematic State

Types describing poses, twists, and joint-space and Cartesian-space states.
See the [Geometry](../guide/geometry.md) chapter of the guide for an introduction.

```{eval-rst}
.. currentmodule:: franky

.. autosummary::
   :toctree: generated
   :nosignatures:

   Affine
   Twist
   TwistAcceleration
   RobotPose
   RobotVelocity
   ElbowState
   FlipDirection
   CartesianState
   CartesianPose
   CartesianVelocities
   JointState
   JointPositions
   JointVelocities
   Torques
   ReferenceType
   CartesianReference
   JointReference
```

## Trajectory Motions

Motion types for the four trajectory control modes.
See the [Motion Types](../guide/motion-types.md) chapter of the guide for an introduction.

### Joint Position Control

```{eval-rst}
.. currentmodule:: franky

.. autosummary::
   :toctree: generated
   :nosignatures:

   Motion
   JointMotion
   JointWaypointMotion
   JointWaypoint
   JointStopMotion
   BaseJointPositionMotion
```

### Joint Velocity Control

```{eval-rst}
.. currentmodule:: franky

.. autosummary::
   :toctree: generated
   :nosignatures:

   JointVelocityMotion
   JointVelocityWaypointMotion
   JointVelocityWaypoint
   JointVelocityStopMotion
   BaseJointVelocityMotion
```

### Cartesian Position Control

```{eval-rst}
.. currentmodule:: franky

.. autosummary::
   :toctree: generated
   :nosignatures:

   CartesianMotion
   CartesianWaypointMotion
   CartesianWaypoint
   CartesianStopMotion
   BaseCartesianPoseMotion
```

### Cartesian Velocity Control

```{eval-rst}
.. currentmodule:: franky

.. autosummary::
   :toctree: generated
   :nosignatures:

   CartesianVelocityMotion
   CartesianVelocityWaypointMotion
   CartesianVelocityWaypoint
   CartesianVelocityStopMotion
   BaseCartesianVelocityMotion
```

## Impedance and Torque Control

Client-side impedance controllers running in torque mode.
See the [Impedance Control](../guide/impedance-control.md) chapter of the guide for an introduction.

```{eval-rst}
.. currentmodule:: franky

.. autosummary::
   :toctree: generated
   :nosignatures:

   BaseTorqueMotion
   ImpedanceMotion
   JointImpedanceBase
   JointImpedanceMotion
   JointImpedanceTrackingMotion
   JointImpedanceTracker
   JointImpedanceGains
   JointImpedanceParams
   CartesianImpedanceBase
   CartesianImpedanceMotion
   CartesianImpedanceTrackingMotion
   CartesianImpedanceTracker
   CartesianImpedanceGains
   CartesianImpedanceParams
   CartesianImpedanceMotionParams
   NullspaceGains
   PostureTask
   ManipulabilityTask
   FrictionCompensationParams
   TorqueSafetyParams
   TorqueStopMotion
   TorqueStopParams
```

## Reactions

Real-time reactions to sensor events.
See the [Real-Time Reactions](../guide/reactions.md) chapter of the guide for an introduction.

```{eval-rst}
.. currentmodule:: franky

.. autosummary::
   :toctree: generated
   :nosignatures:

   Reaction
   Condition
   Measure
   TorqueReaction
   JointPositionReaction
   JointVelocityReaction
   CartesianPoseReaction
   CartesianVelocityReaction
```

## Gripper

Control of the Franka Hand.
See the [Gripper](../guide/gripper.md) chapter of the guide for an introduction.

```{eval-rst}
.. currentmodule:: franky

.. autosummary::
   :toctree: generated
   :nosignatures:

   Gripper
   GripperState
   BoolFuture
```

## Desk Web Interface

Programmatic access to the Franka Desk web interface.
See the [Web Interface API](../guide/desk.md) chapter of the guide for an introduction.

```{eval-rst}
.. currentmodule:: franky

.. autosummary::
   :toctree: generated
   :nosignatures:

   Desk
   DeskWebSession
   BaseDesk
   PilotButton
   PilotButtonEvent
   BrakeState
   OperatingMode
```

## Exceptions

```{eval-rst}
.. currentmodule:: franky

.. autosummary::
   :toctree: generated
   :nosignatures:

   Exception
   CommandException
   ControlException
   GripperException
   IncompatibleVersionException
   InvalidMotionTypeException
   InvalidOperationException
   ModelException
   MotionReuseException
   NetworkException
   ProtocolException
   ReactionRecursionException
   RealtimeException
   DeskError
   FrankaAPIError
   TakeControlTimeoutError
```
