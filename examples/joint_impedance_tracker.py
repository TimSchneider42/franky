from argparse import ArgumentParser

from franky import FrictionCompensationParams, JointImpedanceTracker, Robot


FRICTION_COULOMB = [0.5, 0.4, 0.5, 0.4, 0.4, 0.4, 0.2]
FRICTION_VISCOUS = [0.08, 0.05, 0.08, 0.05, 0.08, 0.08, 0.05]


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
    parser.add_argument(
        "--stiffness",
        type=float,
        default=2.0,
        help="Joint stiffness",
    )
    parser.add_argument(
        "--lock-joint6",
        action="store_true",
        help="Hold joint 6 at its starting angle",
    )
    parser.add_argument(
        "--lock-joint7",
        action="store_true",
        help="Hold joint 7 at its starting angle",
    )
    parser.add_argument(
        "--friction",
        action="store_true",
        help="Enable friction compensation",
    )

    args = parser.parse_args()

    robot = Robot(args.host)
    robot.recover_from_errors()
    lower_joint_limits, upper_joint_limits = get_joint_limits(robot)
    robot.set_collision_behavior(
        torque_thresholds=[35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0],
        force_thresholds=[60.0, 60.0, 60.0, 60.0, 60.0, 60.0],
    )

    q_current = robot.current_joint_positions
    stiffness = [args.stiffness] * 7
    locked_targets = list(q_current)

    if args.lock_joint6:
        stiffness[5] = max(stiffness[5], 40.0)
    if args.lock_joint7:
        stiffness[6] = max(stiffness[6], 40.0)
    print("Entering compliant joint impedance mode.")
    print(
        "You should be able to guide the robot by hand while feeling pushback near joint limits."
    )
    if args.lock_joint6 or args.lock_joint7:
        locked = []
        if args.lock_joint6:
            locked.append("6")
        if args.lock_joint7:
            locked.append("7")
        print(f"Joint lock active on joint(s): {', '.join(locked)}.")
    if args.friction:
        print("Friction compensation enabled.")
    print("Press Ctrl-C to stop.")

    with JointImpedanceTracker(
        robot,
        stiffness=stiffness,
        friction=(
            FrictionCompensationParams(
                coulomb=FRICTION_COULOMB, viscous=FRICTION_VISCOUS
            )
            if args.friction
            else None
        ),
        lower_joint_limits=lower_joint_limits,
        upper_joint_limits=upper_joint_limits,
        period=0.001,
    ) as tracker:
        while tracker.tick():
            q_ref = robot.current_joint_positions
            if args.lock_joint6:
                q_ref[5] = locked_targets[5]
            if args.lock_joint7:
                q_ref[6] = locked_targets[6]
            tracker.set_target(q_ref, dq=[0.0] * 7, tau_ff=[0.0] * 7)
