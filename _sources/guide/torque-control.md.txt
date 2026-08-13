# 💪 Torque Control

The [impedance controllers](impedance-control.md) compute joint torques from a
spring-damper law. If you want to command joint torques directly instead — for
example because you run your own controller, a learned policy, or a teleoperation
scheme — use `SimpleTorqueMotion`. It applies the torques it is given, without any
feedback law on top.

Like the impedance controllers, `SimpleTorqueMotion` keeps the same motion alive
while the commanded torque is updated online. It is therefore started once, and
new torques are published with `set_torque()`:

```python
import time
import numpy as np
from franky import Robot, SimpleTorqueMotion, TorqueStopMotion

robot = Robot("10.90.90.1")

motion = SimpleTorqueMotion()
robot.move(motion, asynchronous=True)

start_time = time.time()
while time.time() - start_time < 5.0:
    # A gentle sinusoidal torque on the last joint.
    torque = np.zeros(7)
    torque[6] = 0.5 * np.sin(2.0 * np.pi * (time.time() - start_time))
    motion.set_torque(torque)
    time.sleep(0.01)

robot.move(TorqueStopMotion())
```

As with the impedance controllers, the commanded torques are added on top of the
robot's internal gravity compensation, so a torque of zero leaves the arm floating.

## The Signal Watchdog

In between two calls to `set_torque()`, the robot keeps applying the last commanded
torque. If the process that produces the torques stalls or dies, the arm would keep
accelerating under a torque that is no longer intended. `SimpleTorqueMotion` therefore
runs a watchdog: if no new torque arrives within `signal_timeout` seconds, the motion
terminates with a `TorqueSignalTimeoutException`.

The default timeout is 50 ms, so `set_torque()` has to be called at least every 50 ms,
even if the torque does not change. As the watchdog is armed the moment the motion
starts, it also bounds how long the initial torque is applied. Loosen it for slower
control loops:

```python
motion = SimpleTorqueMotion(signal_timeout=0.2)
```

or disable it entirely with `signal_timeout=None`, in which case the last commanded
torque is held indefinitely.

Note that the watchdog fires in the robot's control thread, so, like any other error
during an asynchronous motion, the exception is stored and rethrown by the next call
to `Robot.join_motion` or `Robot.move` (see [Real-Time Motions](realtime-motions.md)).

## Initial Torque

Until the first call to `set_torque()`, the motion applies its initial torque, which
is zero by default. Set it to start from something else than a floating arm:

```python
motion = SimpleTorqueMotion(initial_torque=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
```

The initial torque can also be combined with a disabled watchdog to apply a constant
torque without any further signals:

```python
motion = SimpleTorqueMotion(
    initial_torque=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
    signal_timeout=None,
)
robot.move(motion, asynchronous=True)
```

## Safety Limits

`SimpleTorqueMotion` applies the same safety mechanisms as the impedance controllers:

- `max_delta_tau` (1 Nm by default) limits how much the commanded torque may change per
  control cycle, so a large step in the commanded torque is ramped in over multiple
  cycles rather than jerking the arm.
- `lower_joint_limits` and `upper_joint_limits` (unset by default) enable a soft
  joint-limit repulsion that pushes the joints back into their range.
- `friction` accepts `FrictionCompensationParams` to add a friction feedforward term.
- `compensate_coriolis` (off by default here, as the commanded torque is otherwise
  applied verbatim) adds the Coriolis torques of the robot model.

## Ending the Motion

Like the impedance motions, `SimpleTorqueMotion` never finishes on its own. Use
`TorqueStopMotion` to bring the arm to rest and end the control loop cleanly, as
described in [Ending a Torque Motion](impedance-control.md#ending-a-torque-motion).
