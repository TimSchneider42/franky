# 🖥️ Simulating the Robot

Looking to test your code in simulation before moving to the real system?
Don't worry, we got you covered!
[franky-sim](https://github.com/TimSchneider42/franky-sim) is a high-fidelity simulation server for the FR3 that speaks the same network protocol as the real robot and serves as a drop-in replacement.
Hence, you can run the same franky code both in simulation and on the real system!

```{image} ../simulation.webp
:width: 100%
:alt: franky-sim simulation
```

Install it via

```bash
pip install franky-sim
```

Then run a simulation server and connect franky to it:

```python
import franky
from franky_sim import SimulationServer
from franky_sim.mujoco_simulator import MujocoSimulator

with MujocoSimulator(enable_visualization=True) as sim:
    robot_model = sim.add_robot()
    with SimulationServer(sim) as server:
        server.run_async()
        robot = franky.Robot(robot_model.hostname, realtime_config=franky.RealtimeConfig.Ignore)

        robot.move(franky.CartesianMotion(franky.Affine([0.1, 0.0, 0.0]), franky.ReferenceType.Relative))
        print("End-effector pose:", robot.current_cartesian_state.pose.end_effector_pose)
```

See the [franky-sim repository](https://github.com/TimSchneider42/franky-sim) for more examples and configuration options.
