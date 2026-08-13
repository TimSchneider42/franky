# ⏱️ Real-Time Motions

By setting the `asynchronous` parameter of `Robot.move` to `True`, the function does not block until the motion
finishes.
Instead, it returns immediately and, thus, allows the main thread to set new motions asynchronously.

```python
import time
from franky import Affine, CartesianMotion, Robot, ReferenceType

robot = Robot("10.90.90.1")
robot.relative_dynamics_factor = 0.05

motion1 = CartesianMotion(Affine([0.2, 0.0, 0.0]), ReferenceType.Relative)
robot.move(motion1, asynchronous=True)

time.sleep(0.5)
# Note that, similar to reactions, when preempting active motions with new motions, the
# control mode cannot change. Hence, we cannot use, e.g., a JointMotion here.
motion2 = CartesianMotion(Affine([0.2, 0.0, 0.0]), ReferenceType.Relative)
robot.move(motion2, asynchronous=True)
```

By calling `Robot.join_motion`, the main thread can be synchronized with the motion thread, as it will block until the
robot finishes its motion.

```python
robot.join_motion()
```

Note that when exceptions occur during the asynchronous execution of a motion, they will not be thrown immediately.
Instead, the control thread stores the exception and terminates.
The next time `Robot.join_motion` or `Robot.move` is called, they will throw the stored exception in the main thread.
Hence, after an asynchronous motion has finished, make sure to call `Robot.join_motion` to ensure being notified of any
exceptions that occurred during the motion.
