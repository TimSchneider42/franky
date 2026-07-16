# 🫨 Impedance Control

In addition to the [trajectory-based motion types](motion-types.md), franky also provides
client-side impedance controllers in torque mode.

There are two variants for both joint-space and Cartesian impedance:

- `JointImpedanceMotion` and `CartesianImpedanceMotion` are fixed-target motions.
  They interpret a target once at motion start and then regulate toward it.
- `JointImpedanceTrackingMotion` and `CartesianImpedanceTrackingMotion` keep the
  same controller alive while consuming updated references online.

The tracking variants are useful when the desired reference changes every
control cycle, for example for manual guidance, teleoperation, or virtual
fixtures. In Python, the recommended interface for these use cases is
`JointImpedanceTracker` or `CartesianImpedanceTracker`.

When using the tracking motions directly, references and gains are updated at
runtime through `set_reference()` and `set_gains()`, which validate their
inputs and forward them to the control loop. The corresponding
`get_reference()` and `get_gains()` methods return copies of the last
commanded values, so mutating a returned object has no effect until it is
passed back to the setter:

```python
gains = motion.get_gains()
gains.stiffness = [100.0] * 7
motion.set_gains(gains)
```

## Joint Impedance Control

Joint-space impedance can be used either as a fixed posture controller

```python
from franky import JointImpedanceMotion

motion = JointImpedanceMotion(
    target=[0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.7],
    stiffness=[600.0] * 7,
    damping=[50.0] * 7,
)
```

or as a tracking controller with online reference updates:

```python
from franky import JointImpedanceTracker

with JointImpedanceTracker(
    robot,
    stiffness=[600.0] * 7,
    damping=[50.0] * 7,
    period=0.01,
) as tracker:
    while tracker.tick():
        tracker.set_target(
            [0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.7],
            dq=[0.0] * 7,
        )
```

Joint tracking targets can optionally include feedforward torques `tau_ff`. This is
added on top of any `constant_torque_offset` configured on the tracker itself.

All joint-space impedance motions (`JointImpedanceMotion`, `JointImpedanceTrackingMotion`,
and `JointImpedanceTracker`) support optional friction compensation. Coulomb and viscous
terms are added as feedforward terms to the commanded torque each cycle and clamped per joint:

```python
from franky import FrictionCompensationParams

# Use friction compensation to get a smooth zero-g mode kinesthetic demonstrations
with JointImpedanceTracker(
    robot,
    stiffness=[0.0] * 7,
    damping=[0.0] * 7,
    friction=FrictionCompensationParams(
        coulomb=[0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2],   # [Nm]
        viscous=[0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05], # [Nms/rad]
        max_torque=[2.0] * 7,  # per-joint clamp [Nm]
    ),
    period=0.01,
) as tracker:
    ...
```

The Coulomb term uses a smooth sign approximation controlled by
`FrictionCompensationParams.velocity_epsilon` (default `0.03 rad/s`). Either or both
terms can be omitted; unset terms default to zero. `CartesianImpedanceTracker` accepts
the same `friction=` object.

Joint impedance can additionally be shaped in Cartesian space. Passing
`cartesian_stiffness` (a 6-vector `[x, y, z, rx, ry, rz]` at the end effector,
expressed in the base frame) adds `J^T diag(cartesian_stiffness) J` on top of
the joint-space stiffness each cycle, with `cartesian_damping` handled likewise
(critical damping when omitted). This is useful to stay compliant about a joint
posture while stiffening a specific Cartesian direction:

```python
with JointImpedanceTracker(
    robot,
    stiffness=[20.0] * 7,
    cartesian_stiffness=[800.0, 800.0, 0.0, 0.0, 0.0, 0.0],  # stiff in x/y only
    period=0.01,
) as tracker:
    ...
```

Both trackers support updating impedance gains at runtime via `set_gains()`.
Changes are smoothed in the RT loop using exponential interpolation, so abrupt
stiffness steps are avoided automatically.

All torque-mode motions accept a `max_delta_tau` parameter that limits the
commanded torque change per control cycle in Nm, which can help avoid
discontinuity errors from abrupt torque steps.

## Cartesian Impedance Control

Cartesian impedance follows the same split:

```python
from franky import Affine, CartesianImpedanceMotion, ReferenceType

motion = CartesianImpedanceMotion(
    target=Affine([0.45, 0.0, 0.35]),
    target_type=ReferenceType.Absolute,
    translational_stiffness=1200.0,
    rotational_stiffness=80.0,
)
```

```python
from franky import (
    Affine,
    CartesianImpedanceTracker,
    PostureTask,
    Twist,
)

with CartesianImpedanceTracker(
    robot,
    translational_stiffness=1200.0,
    rotational_stiffness=80.0,
    posture_task=PostureTask([0.0, -0.6, 0.0, -2.2, 0.0, 1.7, 0.7], stiffness=10.0),
    max_delta_tau=0.5,
    period=0.01,
) as tracker:
    while tracker.tick():
        tracker.set_target(
            Affine([0.45, 0.0, 0.35]),
            Twist([0.0, 0.0, 0.05], [0.0, 0.0, 0.0]),
        )
```

For Cartesian tracking, the twist argument to `set_target` is optional. When provided, it is
interpreted as the desired end-effector twist in the base frame, so the damping
term acts on twist error instead of damping all motion toward zero.

When a `period` is configured, `tracker.tick()` maintains that loop rate using
`time.perf_counter()` internally and compensates for the time spent in the loop
body. `tracker.elapsed_time` and `tracker.tick_count` are available for
time-based target generation.

By default, Cartesian damping is chosen internally as critically damped with
respect to the requested stiffness. When that is not enough — high-stiffness
configurations often need more — the damping can be set explicitly through
`CartesianImpedanceGains`, either isotropically or per axis:

```python
from franky import CartesianImpedanceGains

# Isotropic stiffness with explicit (over-)damping:
tracker.motion.set_gains(
    CartesianImpedanceGains.isotropic(
        1200.0, 80.0, translational_damping=140.0, rotational_damping=18.0
    )
)

# Or fully anisotropic diagonal gains, ordered [x, y, z, rx, ry, rz]:
tracker.motion.set_gains(
    CartesianImpedanceGains.diagonal(
        [1200, 1200, 1200, 80, 80, 80],
        [140, 140, 140, 18, 18, 18],
    )
)
```

The tracker also exposes damping directly: `tracker.set_gains(damping=[...])` pins a full
6-vector, and `tracker.set_gains(damping=franky.CRITICAL)` unpins it. Damping is all-or-nothing
and omitting it means critical. A stiffness change re-criticals damping unless `damping` is
passed in the same call. `JointImpedanceTracker.set_gains` follows the same rule with 7-vectors.

Cartesian impedance motions also support optional secondary objectives projected
into the Jacobian nullspace, so they bias the redundant arm posture without
changing the Cartesian task to first order. Pass a `posture_task` (a
`PostureTask`) to regulate toward a joint posture, and/or a `manipulability_task`
(a `ManipulabilityTask`) to maximize manipulability; at most one of each is
supported. A `PostureTask`'s per-joint stiffness can weight joints differently or
exclude them (zero stiffness). Keep in mind that the projection spreads the
resulting torque over all joints participating in the self-motion: pushing a
single joint biases the whole null-space posture rather than moving that joint
alone.

```python
from franky import ManipulabilityTask, PostureTask

with CartesianImpedanceTracker(
    robot,
    translational_stiffness=200.0,
    rotational_stiffness=10.0,
    posture_task=PostureTask(robot.current_joint_state.position, stiffness=2.0),
    manipulability_task=ManipulabilityTask(gain=5.0, max_torque=1.0),
    period=0.01,
) as tracker:
    ...
```

The task set is fixed at construction, but the gains can be retuned online:
`tracker.set_gains(posture_stiffness=...)` (scalar or 7-vector) nudges just the
posture stiffness, or `tracker.set_gains(nullspace_gains=NullspaceGains(...))`
replaces the full posture + manipulability gains. Retuning a gain for a task that
was not configured at construction raises.

`CartesianImpedanceTracker` also accepts `translational_error_clip` and
`rotational_error_clip` (each a 3-vector in m and rad respectively) to hard-clip
the pose error fed into the spring law, which can prevent large torque spikes
when the reference jumps.

## Ending a Torque Motion

Unlike the trajectory-based motions, impedance (torque) motions never signal
completion on their own, and `Robot.stop()` preempts the control loop with a
`ControlException` rather than ramping down. To end a torque loop cleanly, use
`TorqueStopMotion`. It blends the last commanded torque into a zero-stiffness
joint-damping law — so it yields to the arm's current pose instead of snapping
to a target — brings the arm to rest, and then finishes. Enqueue it to preempt a
running impedance motion:

```python
from franky import TorqueStopMotion

robot.move(TorqueStopMotion())
```

or attach it as a reaction so the arm damps out automatically once a condition is
met (as with any reaction, the reaction motion must use the same control mode, so
it must itself be a torque motion):

```python
from franky import Measure, Reaction, TorqueStopMotion

motion.add_reaction(Reaction(Measure.FORCE_Z > 20.0, TorqueStopMotion()))
```

`TorqueStopMotion` also finishes after `max_duration` regardless of velocity, so
a sustained external push can never make the stop hang.
