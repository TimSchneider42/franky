# ⚡ Real-Time Reactions

By adding reactions to the motion data, the robot can react to unforeseen events.
In the Python API, you can define conditions by using a comparison between a robot's value and a given threshold.
If the threshold is exceeded, the reaction fires.

```python
from franky import CartesianMotion, Affine, ReferenceType, Measure, Reaction

motion = CartesianMotion(Affine([0.0, 0.0, 0.1]), ReferenceType.Relative)  # Move down 10cm

# It is important that the reaction motion uses the same control mode as the original motion.
# Hence, we cannot register a JointMotion as a reaction motion to a CartesianMotion.
# Move up by 1cm
reaction_motion = CartesianMotion(Affine([0.0, 0.0, -0.01]), ReferenceType.Relative)

# Trigger reaction if the Z force is greater than 30N
reaction = Reaction(Measure.FORCE_Z > 5.0, reaction_motion)
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

In C++, you can additionally use lambdas to define more complex behaviours:

```c++
auto motion = CartesianMotion(
  RobotPose(Affine({0.0, 0.0, 0.2}), 0.0), ReferenceType::Relative);

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
