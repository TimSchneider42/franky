from argparse import ArgumentParser

from franky import (
    Affine,
    JointWaypointMotion,
    JointWaypoint,
    Robot,
    CartesianWaypointMotion,
    CartesianWaypoint,
    ReferenceType,
    RobotPose,
    ElbowState,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")
    args = parser.parse_args()

    # Connect to the robot
    robot = Robot(args.host)
    robot.recover_from_errors()

    # Reduce the acceleration and velocity dynamic
    robot.relative_dynamics_factor = 0.1

    joint_motion = JointWaypointMotion(
        [
            JointWaypoint([0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7]),
            JointWaypoint([0.2, -0.1, 0.3, -2.0, 0.1, 1.9, 0.6]),
            JointWaypoint([0.4, -0.3, 0.1, -2.3, 0.0, 2.2, 0.5]),
        ]
    )
    robot.move(joint_motion)

    # Define and move forwards
    wp_motion = CartesianWaypointMotion(
        [
            CartesianWaypoint(
                RobotPose(Affine([0.0, 0.0, -0.12]), ElbowState(-0.2)),
                ReferenceType.Relative,
            ),
            CartesianWaypoint(
                RobotPose(Affine([0.08, 0.0, 0.0]), ElbowState(0.0)),
                ReferenceType.Relative,
            ),
            CartesianWaypoint(
                RobotPose(Affine([0.0, 0.1, 0.0]), ElbowState(0.0)),
                ReferenceType.Relative,
            ),
        ]
    )

    # You can try to block the robot now.
    robot.move(wp_motion)
