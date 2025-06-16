from argparse import ArgumentParser

from franky import (
    Affine,
    JointMotion,
    Measure,
    Reaction,
    Robot,
    CartesianStopMotion,
    CartesianMotion,
    RobotPose,
    RobotState,
    ReferenceType,
)


def reaction_callback(robot_state: RobotState, rel_time: float, abs_time: float):
    print(f"Robot stopped at time {rel_time}.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")
    args = parser.parse_args()

    # Connect to the robot
    robot = Robot(args.host)
    robot.recover_from_errors()

    # Reduce the acceleration and velocity dynamic
    robot.relative_dynamics_factor = 0.1

    # Go to initial position
    robot.move(JointMotion([0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7]))

    # Define and move forwards
    reaction = Reaction(Measure.FORCE_Z < -5.0, CartesianStopMotion())
    reaction.register_callback(reaction_callback)
    motion_down = CartesianMotion(
        RobotPose(Affine([0.0, 0.0, 0.5])), ReferenceType.Relative
    )
    motion_down.add_reaction(reaction)

    # You can try to block the robot now.
    robot.move(motion_down)
