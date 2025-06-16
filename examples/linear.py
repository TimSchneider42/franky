from argparse import ArgumentParser

from franky import Affine, CartesianMotion, Robot, ReferenceType, JointMotion


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
    target = Affine([0.0, 0.2, 0.0])
    motion_forward = CartesianMotion(target, reference_type=ReferenceType.Relative)
    robot.move(motion_forward)

    # And move backwards using the inverse motion
    motion_backward = CartesianMotion(
        target.inverse, reference_type=ReferenceType.Relative
    )
    robot.move(motion_backward)
