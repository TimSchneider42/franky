# 🎬 Executing Motions

## Relative Dynamics Factors

Every motion and waypoint type allows for adapting the dynamics (velocity, acceleration, and jerk) by setting the respective
`relative_dynamics_factor` parameter.
This parameter can also be set for the robot globally, as shown below, or in the `robot.move` command.
Crucially, relative dynamics factors on different layers (robot, move command, and motion) do not override each other
but rather get multiplied.
Hence, a relative dynamics factor on a motion can only reduce the dynamics of the robot and never increase them.

There is one exception to this rule, and that is if any layer sets the relative dynamics factor to
`RelativeDynamicsFactor.MAX_DYNAMICS`.
This will cause the motion to be executed with maximum velocity, acceleration, and jerk limits, independently of the
relative dynamics factors of the other layers.
This feature should only be used to abruptly stop the robot in case of an unexpected environment contact, as executing
Other motions with it are likely to lead to a discontinuity error and might be dangerous.

## The Move Command

The real robot can be moved by applying a motion to the robot using `move`:

```python
# Before moving the robot, set an appropriate dynamics factor. We start small:
robot.relative_dynamics_factor = 0.05
# or alternatively, to control the scaling of velocity, acceleration, and jerk limits
# separately:
robot.relative_dynamics_factor = RelativeDynamicsFactor(0.05, 0.1, 0.15)
# If these values are set too high, you will see discontinuity errors

robot.move(m_jp1)

# We can also set a relative dynamics factor in the move command. It will be multiplied by
# the other relative dynamics factors (robot and motion if present).
robot.move(m_jp2, relative_dynamics_factor=0.8)
```

Note that motion objects are single-use: they maintain internal state, so once a motion has been started, passing it
to `move` again (or returning it from a reaction) raises a `MotionReuseException`.
Create a new motion instance for every execution instead.

## Motion Callbacks

All motions support callbacks, which will be invoked in every control step at 1kHz.
Callbacks can be attached as follows:

```python
def cb(
        robot_state: RobotState,
        time_step: Duration,
        rel_time: Duration,
        abs_time: Duration,
        control_signal: JointPositions,
):
    print(f"At time {abs_time}, the target joint positions were {control_signal.q}")


m_cb = JointMotion([-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0.7])
m_cb.register_callback(cb)
robot.move(m_cb)
```

Note that in Python, these callbacks are not executed in the control thread since they would otherwise block it.
Instead, they are put in a queue and executed by another thread.
While this scheme ensures that the control thread can always run, it cannot prevent the queue from growing indefinitely
when the callbacks take more time to execute than it takes for new callbacks to be queued.
Hence, callbacks might be executed significantly after they were queued if they take a long time to execute.
