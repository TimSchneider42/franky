from argparse import ArgumentParser

from franky import JointImpedanceMotion, Robot


# Franka Panda / FR3 joint limits from the standard libfranka model.
DEFAULT_LOWER_JOINT_LIMITS = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
DEFAULT_UPPER_JOINT_LIMITS = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")
    parser.add_argument(
        "--stiffness",
        type=float,
        default=6.0,
        help="Uniform joint stiffness used for manual guidance",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=1.0,
        help="Uniform joint damping used for manual guidance",
    )
    parser.add_argument(
        "--activation-distance",
        type=float,
        default=0.15,
        help="Distance from a joint limit where pushback starts [rad]",
    )
    parser.add_argument(
        "--limit-stiffness",
        type=float,
        default=4.0,
        help="Repulsion gain for the joint-limit soft wall [Nm]",
    )
    parser.add_argument(
        "--limit-damping",
        type=float,
        default=1.0,
        help="Extra damping when moving into a joint limit [Nms/rad]",
    )
    parser.add_argument(
        "--limit-max-torque",
        type=float,
        default=5.0,
        help="Maximum absolute torque contributed by the joint-limit soft wall [Nm]",
    )
    args = parser.parse_args()

    robot = Robot(args.host)
    robot.recover_from_errors()

    q_current = robot.current_joint_positions
    stiffness = [args.stiffness] * 7
    damping = [args.damping] * 7

    motion = JointImpedanceMotion(
        target=q_current,
        stiffness=stiffness,
        damping=damping,
        lower_joint_limits=DEFAULT_LOWER_JOINT_LIMITS,
        upper_joint_limits=DEFAULT_UPPER_JOINT_LIMITS,
        joint_limit_activation_distance=args.activation_distance,
        joint_limit_stiffness=args.limit_stiffness,
        joint_limit_damping=args.limit_damping,
        joint_limit_max_torque=args.limit_max_torque,
        compensate_coriolis=True,
        max_delta_tau=1.0,
    )

    print("Entering compliant joint impedance mode.")
    print("You should be able to guide the robot by hand while feeling pushback near joint limits.")
    print("Press Ctrl-C to stop.")

    robot.move(motion, asynchronous=True)

    try:
        robot.join_motion()
    except KeyboardInterrupt:
        print("\nStopping...")
        robot.stop()
        robot.join_motion()
