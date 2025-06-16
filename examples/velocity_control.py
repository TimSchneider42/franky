import time
from argparse import ArgumentParser

from franky._franky import RelativeDynamicsFactor

from franky import (
    Robot,
    JointVelocityMotion,
    CartesianVelocityMotion,
    Duration,
    JointMotion,
    Twist,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")
    args = parser.parse_args()

    # Connect to the robot
    robot = Robot(args.host)
    robot.recover_from_errors()

    # Reduce the acceleration and velocity dynamic
    robot.relative_dynamics_factor = RelativeDynamicsFactor(0.1, 0.05, 0.05)

    # Go to initial position
    robot.move(JointMotion([0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7]))

    time.sleep(1.0)

    print("Starting joint velocity control...")

    # Reach the given joint velocities and hold them for 3000ms
    robot.move(
        JointVelocityMotion(
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], duration=Duration(3000)
        )
    )

    print("Joint velocity control stopped.")

    time.sleep(1.0)

    # Back to initial position
    robot.move(JointMotion([0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7]))

    print("Starting cartesian velocity control...")

    time.sleep(1.0)

    # Reach the given twist (cartesian velocity) and hold it for 3000ms
    robot.move(
        CartesianVelocityMotion(
            Twist(
                linear_velocity=[0.03, 0.03, 0.03], angular_velocity=[0.03, 0.03, 0.03]
            ),
            duration=Duration(3000),
        )
    )

    print("Cartesian velocity control stopped.")
