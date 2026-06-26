from argparse import ArgumentParser

import numpy as np

from franky import Affine, CartesianImpedanceTracker, Robot, Twist


def get_joint_limits(robot: Robot):
    if "fr3" in robot.model_urdf.lower():
        return (
            [-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508],
            [2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508],
        )
    return (
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")
    args = parser.parse_args()

    direction = np.array([0.0, 0.0, 1.0], dtype=float)

    robot = Robot(args.host)
    robot.recover_from_errors()
    robot.set_collision_behavior(
        torque_thresholds=[35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0],
        force_thresholds=[60.0, 60.0, 60.0, 60.0, 60.0, 60.0],
    )
    lower_joint_limits, upper_joint_limits = get_joint_limits(robot)

    current_pose = robot.current_pose.end_effector_pose
    origin = np.asarray(current_pose.translation, dtype=float)
    orientation = np.asarray(current_pose.quaternion, dtype=float)

    print("Entering Cartesian virtual-fixture mode.")
    print(
        "The robot will feel compliant along the world Z axis and resist motion away from it."
    )
    print("Press Ctrl-C to stop.")

    with CartesianImpedanceTracker(
        robot,
        translational_stiffness=800,
        rotational_stiffness=60,
        lower_joint_limits=lower_joint_limits,
        upper_joint_limits=upper_joint_limits,
        period=0.001,
    ) as tracker:
        previous_projected = origin.copy()
        while tracker.tick():
            pose = tracker.current_pose.end_effector_pose
            position = np.asarray(pose.translation, dtype=float)
            signed_distance = float(np.dot(position - origin, direction))
            projected = origin + signed_distance * direction
            desired_linear_velocity = (projected - previous_projected) / 0.001
            tracker.set_target(Affine(projected, orientation))
            previous_projected = projected
