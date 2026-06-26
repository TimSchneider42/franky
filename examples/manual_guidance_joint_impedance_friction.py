[33mcommit 57f06426b79e14e41f9e486d04ae533e189e667f[m
Author: Nick Walker <nick@nickwalker.us>
Date:   Thu Jun 25 20:11:11 2026 -0400

    Add joint impedance friction compensation
    
    Add optional Coulomb and viscous friction compensation to the joint
    impedance controller.  Per-joint gains and a smooth tanh sign
    transition (velocity_epsilon) avoid torque discontinuities near zero
    velocity.  The Python tracker exposes the params directly via the
    constructor.
    
    Also pick up missed content from the hardening step: input validation
    helpers (validateNonNegativeFinite, defaultJointImpedanceDamping) in
    the Python bindings, and the atomic terminate flag in SequentialExecutor
    that forces visibility across the destructor/executor thread boundary.
    
    Co-authored-by: Yuqian Jiang <jiangyuqian@utexas.edu>

[33mdiff --git a/examples/manual_guidance_joint_impedance_friction.py b/examples/manual_guidance_joint_impedance_friction.py[m
[33mnew file mode 100644[m
[33mindex 000000000000..ca4799f7bbdb[m
[33m--- /dev/null[m
[33m+++ b/examples/manual_guidance_joint_impedance_friction.py[m
[35m@@ -0,0 +1,214 @@[m
[32m+[m[32mfrom argparse import ArgumentParser[m
[32m+[m[32mimport time[m
[32m+[m
[32m+[m[32mimport numpy as np[m
[32m+[m
[32m+[m[32mfrom franky import JointImpedanceTracker, Robot[m
[32m+[m
[32m+[m
[32m+[m[32m# Franka Panda / FR3 joint limits from the standard libfranka model.[m
[32m+[m[32mDEFAULT_LOWER_JOINT_LIMITS = [[m
[32m+[m[32m    -2.8973,[m
[32m+[m[32m    -1.7628,[m
[32m+[m[32m    -2.8973,[m
[32m+[m[32m    -3.0718,[m
[32m+[m[32m    -2.8973,[m
[32m+[m[32m    -0.0175,[m
[32m+[m[32m    -2.8973,[m
[32m+[m[32m][m
[32m+[m[32mDEFAULT_UPPER_JOINT_LIMITS = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973][m
[32m+[m
[32m+[m[32mDEFAULT_FRICTION_ENABLED = [1, 1, 1, 1, 1, 1, 1][m
[32m+[m[32mDEFAULT_FRICTION_COULOMB = [0.5, 0.4, 0.5, 0.4, 0.4, 0.4, 0.2][m
[32m+[m[32mDEFAULT_FRICTION_VISCOUS = [0.08, 0.05, 0.08, 0.05, 0.08, 0.08, 0.05][m
[32m+[m[32mDEFAULT_FRICTION_LIMIT = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0][m
[32m+[m
[32m+[m
[32m+[m[32mdef friction_compensation(dq, enabled, coulomb, viscous, torque_limit, epsilon):[m
[32m+[m[32m    dq = np.asarray(dq, dtype=float)[m
[32m+[m[32m    enabled = np.asarray(enabled, dtype=bool)[m
[32m+[m[32m    coulomb = np.asarray(coulomb, dtype=float)[m
[32m+[m[32m    viscous = np.asarray(viscous, dtype=float)[m
[32m+[m[32m    torque_limit = np.asarray(torque_limit, dtype=float)[m
[32m+[m
[32m+[m[32m    tau = coulomb * np.tanh(dq / epsilon) + viscous * dq[m
[32m+[m[32m    tau = np.clip(tau, -torque_limit, torque_limit)[m
[32m+[m[32m    return np.where(enabled, tau, 0.0)[m
[32m+[m
[32m+[m
[32m+[m[32mdef format_joint_vector(values):[m
[32m+[m[32m    return "[" + ", ".join(f"{value:+.4f}" for value in values) + "]"[m
[32m+[m
[32m+[m
[32m+[m[32mif __name__ == "__main__":[m
[32m+[m[32m    parser = ArgumentParser()[m
[32m+[m[32m    parser.add_argument("--host", default="172.16.0.2", help="FCI IP of the robot")[m
[32m+[m[32m    parser.add_argument([m
[32m+[m[32m        "--stiffness",[m
[32m+[m[32m        type=float,[m
[32m+[m[32m        default=2.0,[m
[32m+[m[32m        help="Joint stiffness",[m
[32m+[m[32m    )[m
[32m+[m[32m    parser.add_argument([m
[32m+[m[32m        "--lock-joint6",[m
[32m+[m[32m        action="store_true",[m
[32m+[m[32m        help="Hold joint 6 at its starting angle",[m
[32m+[m[32m    )[m
[32m+[m[32m    parser.add_argument([m
[32m+[m[32m        "--lock-joint7",[m
[32m+[m[32m        action="store_true",[m
[32m+[m[32m        help="Hold joint 7 at its starting angle",[m
[32m+[m[32m    )[m
[32m+[m[32m    parser.add_argument([m
[32m+[m[32m        "--lock-stiffness",[m
[32m+[m[32m        type=float,[m
[32m+[m[32m        default=40.0,[m
[32m+[m[32m        help="Stiffness used for locked joints [Nm/rad]",[m
[32m+[m[32m    )[m
[32m+[m[32m    parser.add_argument([m
[32m+[m[32m        "--friction-enabled",[m
[32m+[m[32m        type=int,[m
[32m+[m[32m        nargs=7,[m
[32m+[m[32m        default=DEFAULT_FRICTION_ENABLED,[m
[32m+[m[32m        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),[m
[32m+[m[32m        help="Per-joint friction compensation mask, using 0 or 1",[m
[32m+[m[32m    )[m
[32m+[m[32m    parser.add_argument([m
[32m+[m[32m        "--friction-coulomb",[m
[32m+[m[32m        type=float,[m
[32m+[m[32m        nargs=7,[m
[32m+[m[32m        default=DEFAULT_FRICTION_COULOMB,[m
[32m+[m[32m        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),[m
[32m+[m[32m        help="Per-joint Coulomb friction compensation [Nm]",[m
[32m+[m[32m    )[m
[32m+[m[32m    parser.add_argument([m
[32m+[m[32m        "--friction-viscous",[m
[32m+[m[32m        type=float,[m
[32m+[m[32m        nargs=7,[m
[32m+[m[32m        default=DEFAULT_FRICTION_VISCOUS,[m
[32m+[m[32m        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),[m
[32m+[m[32m        help="Per-joint viscous friction compensation [Nms/rad]",[m
[32m+[m[32m    )[m
[32m+[m[32m    parser.add_argument([m
[32m+[m[32m        "--friction-limit",[m
[32m+[m[32m        type=float,[m
[32m+[m[32m        nargs=7,[m
[32m+[m[32m        default=DEFAULT_FRICTION_LIMIT,[m
[32m+[m[32m        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),[m
[32m+[m[32m        help="Per-joint absolute friction compensation torque limit [Nm]",[m
[32m+[m[32m    )[m
[32m+[m[32m    parser.add_argument([m
[32m+[m[32m        "--friction-mode",[m
[32m+[m[32m        choices=("cpp", "python", "off"),[m
[32m+[m[32m        default="cpp",[m
[32m+[m[32m        help="Where to apply friction compensation",[m
[32m+[m[32m    )[m
[32m+[m[32m    parser.add_argument([m
[32m+[m[32m        "--friction-epsilon",[m
[32m+[m[32m        type=float,[m
[32m+[m[32m        default=0.03,[m
[32m+[m[32m        help="Velocity scale for tanh smoothing [rad/s]",[m
[32m+[m[32m    )[m
[32m+[m[32m    parser.add_argument([m
[32m+[m[32m        "--print-period",[m
[32m+[m[32m        type=float,[m
[32m+[m[32m        default=0.5,[m
[32m+[m[32m        help="Seconds between dq/tau_ff diagnostic prints",[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    args = parser.parse_args()[m
[32m+[m
[32m+[m[32m    if args.friction_epsilon <= 0.0:[m
[32m+[m[32m        raise ValueError("--friction-epsilon must be positive")[m
[32m+[m[32m    if args.print_period < 0.0:[m
[32m+[m[32m        raise ValueError("--print-period must be non-negative")[m
[32m+[m[32m    if any(value not in (0, 1) for value in args.friction_enabled):[m
[32m+[m[32m        raise ValueError("--friction-enabled values must be 0 or 1")[m
[32m+[m[32m    if any(value < 0.0 for value in args.friction_limit):[m
[32m+[m[32m        raise ValueError("--friction-limit values must be non-negative")[m
[32m+[m
[32m+[m[32m    robot = Robot(args.host)[m
[32m+[m[32m    robot.recover_from_errors()[m
[32m+[m[32m    robot.set_collision_behavior([m
[32m+[m[32m        torque_thresholds=[35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0],[m
[32m+[m[32m        force_thresholds=[60.0, 60.0, 60.0, 60.0, 60.0, 60.0],[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    q_current = robot.current_joint_positions[m
[32m+[m[32m    stiffness = [args.stiffness] * 7[m
[32m+[m[32m    locked_targets = list(q_current)[m
[32m+[m
[32m+[m[32m    if args.lock_joint6:[m
[32m+[m[32m        stiffness[5] = max(stiffness[5], args.lock_stiffness)[m
[32m+[m[32m    if args.lock_joint7:[m
[32m+[m[32m        stiffness[6] = max(stiffness[6], args.lock_stiffness)[m
[32m+[m[32m    print("Entering compliant joint impedance mode with friction compensation.")[m
[32m+[m[32m    print([m
[32m+[m[32m        "You should be able to guide the robot by hand while feeling pushback near joint limits."[m
[32m+[m[32m    )[m
[32m+[m[32m    print(f"Friction mode: {args.friction_mode}")[m
[32m+[m[32m    print(f"Friction mask: {args.friction_enabled}")[m
[32m+[m[32m    print(f"Friction Coulomb [Nm]: {format_joint_vector(args.friction_coulomb)}")[m
[32m+[m[32m    print(f"Friction viscous [Nms/rad]: {format_joint_vector(args.friction_viscous)}")[m
[32m+[m[32m    print(f"Friction torque limits [Nm]: {format_joint_vector(args.friction_limit)}")[m
[32m+[m[32m    if args.lock_joint6 or args.lock_joint7:[m
[32m+[m[32m        locked = [][m
[32m+[m[32m        if args.lock_joint6:[m
[32m+[m[32m            locked.append("6")[m
[32m+[m[32m        if args.lock_joint7:[m
[32m+[m[32m            locked.append("7")[m
[32m+[m[32m        print(f"Joint lock active on joint(s): {', '.join(locked)}.")[m
[32m+[m[32m    print("Press Ctrl-C to stop.")[m
[32m+[m
[32m+[m[32m    friction_enabled = np.asarray(args.friction_enabled, dtype=bool)[m
[32m+[m[32m    friction_coulomb = np.asarray(args.friction_coulomb, dtype=float)[m
[32m+[m[32m    friction_viscous = np.asarray(args.friction_viscous, dtype=float)[m
[32m+[m[32m    friction_limit = np.asarray(args.friction_limit, dtype=float)[m
[32m+[m[32m    next_print_time = time.perf_counter()[m
[32m+[m
[32m+[m[32m    with JointImpedanceTracker([m
[32m+[m[32m        robot,[m
[32m+[m[32m        stiffness=stiffness,[m
[32m+[m[32m        compensate_friction=args.friction_mode == "cpp",[m
[32m+[m[32m        friction_coulomb=friction_enabled * friction_coulomb,[m
[32m+[m[32m        friction_viscous=friction_enabled * friction_viscous,[m
[32m+[m[32m        friction_max_torque=friction_enabled * friction_limit,[m
[32m+[m[32m        friction_velocity_epsilon=args.friction_epsilon,[m
[32m+[m[32m        lower_joint_limits=DEFAULT_LOWER_JOINT_LIMITS,[m
[32m+[m[32m        upper_joint_limits=DEFAULT_UPPER_JOINT_LIMITS,[m
[32m+[m[32m        period=0.001,[m
[32m+[m[32m    ) as tracker:[m
[32m+[m[32m        while tracker.tick():[m
[32m+[m[32m            state = tracker.state[m
[32m+[m[32m            q_ref = list(state.q)[m
[32m+[m[32m            if args.lock_joint6:[m
[32m+[m[32m                q_ref[5] = locked_targets[5][m
[32m+[m[32m            if args.lock_joint7:[m
[32m+[m[32m                q_ref[6] = locked_targets[6][m
[32m+[m
[32m+[m[32m            if args.friction_mode == "off":[m
[32m+[m[32m                friction_preview = np.zeros(7)[m
[32m+[m[32m            else:[m
[32m+[m[32m                friction_preview = friction_compensation([m
[32m+[m[32m                    state.dq,[m
[32m+[m[32m                    friction_enabled,[m
[32m+[m[32m                    friction_coulomb,[m
[32m+[m[32m                    friction_viscous,[m
[32m+[m[32m                    friction_limit,[m
[32m+[m[32m                    args.friction_epsilon,[m
[32m+[m[32m                )[m
[32m+[m
[32m+[m[32m            if args.friction_mode == "python":[m
[32m+[m[32m                tau_ff = friction_preview[m
[32m+[m[32m            else:[m
[32m+[m[32m                tau_ff = np.zeros(7)[m
[32m+[m[32m            tracker.set_target(q_ref, dq=[0.0] * 7, tau_ff=tau_ff)[m
[32m+[m
[32m+[m[32m            now = time.perf_counter()[m
[32m+[m[32m            if args.print_period == 0.0 or now >= next_print_time:[m
[32m+[m[32m                print([m
[32m+[m[32m                    f"dq={format_joint_vector(state.dq)} "[m
[32m+[m[32m                    f"friction={format_joint_vector(friction_preview)} "[m
[32m+[m[32m                    f"tau_ff={format_joint_vector(tau_ff)}"[m
[32m+[m[32m                )[m
[32m+[m[32m                next_print_time = now + args.print_period[m
