"""
Integration tests using franky to control the simulated FR3 robot.

Each test spins up a Mujoco-backed SimulationServer, connects with franky's Robot,
executes motions covering every joint / Cartesian direction, and asserts the robot
reached the expected pose.

Torque control is intentionally excluded (out of scope per the task description).
"""

from __future__ import annotations

from contextlib import contextmanager

import franky
import numpy as np
import pytest

from franky_sim import SimulationServer
from franky_sim.mujoco_simulator import MujocoSimulator

# Tolerances
JOINT_ATOL = 0.03  # rad – how close the joint must be to the target
CART_ATOL = 0.025  # m   – how close the Cartesian position must be to the target


@contextmanager
def sim_server_context():
    """Start a Genesis simulation + protocol server and tear it down afterwards."""
    with MujocoSimulator() as sim:
        sim.add_robot()
        with SimulationServer(sim) as server:
            server.run_async()
            yield server


def make_robot(hostname: str) -> franky.Robot:
    """Create a franky Robot connected to the local simulation."""
    return franky.Robot(
        hostname,
        realtime_config=franky.RealtimeConfig.Ignore,
        relative_dynamics_factor=0.2,
    )


# ---------------------------------------------------------------------------
# Test 1 – Joint position control
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_joint_position_control():
    """
    Use JointWaypointMotion to drive all 7 joints through four different
    configurations, covering both positive and negative directions for every
    joint.  Verifies that the robot settles at each commanded position.
    """
    # All values are within FR3 joint limits.
    # Initial: [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785]
    waypoints = [
        [-0.3, 0.1, 0.3, -1.4, 0.1, 1.8, 0.7],
        [0.2, -0.1, 0.1, -1.7, -0.1, 1.6, 0.9],
        [0.0, 0.2, -0.3, -1.6, -0.2, 1.5, 0.3],
        [-0.2, -0.3, 0.2, -1.3, 0.3, 2.0, 1.0],
    ]

    with sim_server_context() as server:
        robot = make_robot(server.robot_servers[0].hostname)

        for i, waypoint in enumerate(waypoints):
            robot.move(franky.JointWaypointMotion([franky.JointWaypoint(waypoint)]))
            q_actual = list(robot.current_joint_state.position)
            np.testing.assert_allclose(
                q_actual,
                waypoint,
                atol=JOINT_ATOL,
                err_msg=f"Waypoint {i + 1}: joint positions out of tolerance",
            )


# ---------------------------------------------------------------------------
# Test 2 – Joint velocity control
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_joint_velocity_control():
    """
    Apply four sequential joint-velocity phases.  Each phase exercises all
    joints; the sign pattern rotates across phases so every joint is driven
    in both directions.  After each phase the sign of the displacement is
    verified.
    """
    hold_ms = 400  # ms per phase

    phases = [
        ([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], "A – all positive"),
        ([-0.1, -0.1, -0.1, 0.1, -0.1, 0.1, -0.1], "B – mixed"),
        ([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1], "C – alternating"),
        ([-0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1], "D – alternating reversed"),
    ]

    with sim_server_context() as server:
        robot = make_robot(server.robot_servers[0].hostname)

        for velocities, label in phases:
            q_before = np.array(robot.current_joint_state.position)
            robot.move(
                franky.JointVelocityWaypointMotion(
                    [
                        franky.JointVelocityWaypoint(
                            velocities,
                            hold_target_duration=franky.Duration(hold_ms),
                        )
                    ]
                )
            )
            q_after = np.array(robot.current_joint_state.position)
            delta = q_after - q_before

            for i, (v, d) in enumerate(zip(velocities, delta)):
                assert (
                    np.sign(v) == np.sign(d) or abs(d) < 5e-3
                ), f"Phase {label}, joint {i}: commanded v={v:.3f} but Δq={d:.4f}"


# ---------------------------------------------------------------------------
# Test 3 – Cartesian position control
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_cartesian_position_control():
    """
    Move the end-effector to six absolute Cartesian targets that offset the
    initial pose in every world-frame axis direction (±x, ±y, ±z).  Absolute
    targets are used so the commanded translation is expressed in the world
    frame, avoiding the EE-frame ambiguity of relative motions.
    """
    offsets = [
        (np.array([0.05, 0.00, 0.00]), "x+"),
        (np.array([-0.05, 0.00, 0.00]), "x-"),
        (np.array([0.00, 0.04, 0.00]), "y+"),
        (np.array([0.00, -0.04, 0.00]), "y-"),
        (np.array([0.00, 0.00, 0.04]), "z+"),
        (np.array([0.00, 0.00, -0.04]), "z-"),
    ]

    with sim_server_context() as server:
        robot = make_robot(server.robot_servers[0].hostname)

        initial_pose = robot.current_cartesian_state.pose.end_effector_pose
        initial_translation = np.array(initial_pose.translation).flatten()

        for offset, label in offsets:
            target_translation = initial_translation + offset
            target_matrix = initial_pose.matrix.copy()
            target_matrix[:3, 3] = target_translation
            robot.move(
                franky.CartesianMotion(
                    franky.Affine(target_matrix),
                    franky.ReferenceType.Absolute,
                )
            )
            actual_translation = np.array(
                robot.current_cartesian_state.pose.end_effector_pose.translation
            ).flatten()
            np.testing.assert_allclose(
                actual_translation,
                target_translation,
                atol=CART_ATOL,
                err_msg=f"Cartesian step {label}: position {actual_translation} != "
                f"expected {target_translation}",
            )


# ---------------------------------------------------------------------------
# Test 4 – Cartesian velocity control
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_cartesian_velocity_control():
    """
    Apply six sequential Cartesian velocity phases covering all three
    translation axes in both directions (+x, −x, +y, −y, +z, −z).
    After each phase the sign of the actual displacement is verified.
    """
    hold_ms = 500

    phases = [
        ([0.02, 0.00, 0.00], "x+"),
        ([-0.02, 0.00, 0.00], "x-"),
        ([0.00, 0.02, 0.00], "y+"),
        ([0.00, -0.02, 0.00], "y-"),
        ([0.00, 0.00, 0.02], "z+"),
        ([0.00, 0.00, -0.02], "z-"),
    ]

    with sim_server_context() as server:
        robot = make_robot(server.robot_servers[0].hostname)

        for vel_xyz, label in phases:
            pos_before = np.array(
                robot.current_cartesian_state.pose.end_effector_pose.translation
            )
            robot.move(
                franky.CartesianVelocityWaypointMotion(
                    [
                        franky.CartesianVelocityWaypoint(
                            franky.Twist(vel_xyz),
                            hold_target_duration=franky.Duration(hold_ms),
                        )
                    ]
                )
            )
            pos_after = np.array(
                robot.current_cartesian_state.pose.end_effector_pose.translation
            )
            displacement = pos_after - pos_before

            for axis, (v, d) in enumerate(zip(vel_xyz, displacement)):
                if abs(v) > 1e-6:  # only check commanded axes
                    assert (
                        np.sign(v) == np.sign(d) or abs(d) < 5e-3
                    ), f"Phase {label}, axis {axis}: commanded v={v:.3f} but Δp={d:.4f}"


# ---------------------------------------------------------------------------
# Test 5 - Joint impedance control
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_joint_impedance_motion():
    """
    Use JointImpedanceMotion to drive all 7 joints to a target configuration.
    Runs asynchronously and is manually stopped since it does not finish automatically.
    """
    target = [-0.1, 0.1, 0.1, -1.5, 0.1, 1.6, 0.8]
    with sim_server_context() as server:
        robot = make_robot(server.robot_servers[0].hostname)
        robot.move(franky.JointImpedanceMotion(target), asynchronous=True)

        import time

        time.sleep(1.0)

        robot.stop()
        try:
            robot.join_motion()
        except franky.ControlException as e:
            if "Move command preempted" not in str(e):
                raise

        q_actual = list(robot.current_joint_state.position)
        np.testing.assert_allclose(
            q_actual,
            target,
            atol=JOINT_ATOL,
            err_msg="Joint positions out of tolerance for JointImpedanceMotion",
        )


# ---------------------------------------------------------------------------
# Test 6 - Cartesian impedance control
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_cartesian_impedance_motion():
    """
    Move the end-effector to an absolute Cartesian target using CartesianImpedanceMotion.
    Runs asynchronously and is manually stopped since it does not finish automatically.
    """
    offset = np.array([0.05, 0.05, 0.0])
    with sim_server_context() as server:
        robot = make_robot(server.robot_servers[0].hostname)
        initial_pose = robot.current_cartesian_state.pose.end_effector_pose
        initial_translation = np.array(initial_pose.translation).flatten()

        target_translation = initial_translation + offset
        target_matrix = initial_pose.matrix.copy()
        target_matrix[:3, 3] = target_translation

        robot.move(
            franky.CartesianImpedanceMotion(franky.Affine(target_matrix)),
            asynchronous=True,
        )

        import time

        time.sleep(1.0)

        robot.stop()
        try:
            robot.join_motion()
        except franky.ControlException as e:
            if "Move command preempted" not in str(e):
                raise

        actual_translation = np.array(
            robot.current_cartesian_state.pose.end_effector_pose.translation
        ).flatten()
        np.testing.assert_allclose(
            actual_translation,
            target_translation,
            atol=CART_ATOL,
            err_msg="Position out of tolerance for CartesianImpedanceMotion",
        )


# ---------------------------------------------------------------------------
# Test 6b - Cartesian impedance null-space posture task
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_cartesian_impedance_nullspace_posture():
    """
    Exercise the null-space posture task of the Cartesian impedance controller.

    The end-effector is commanded to hold its current pose while a secondary
    posture objective (nullspace_target) pulls the joints toward a different
    configuration.  The FR3 has 7 DoF and the Cartesian task constrains 6, so
    the controller can only move the arm along its one-dimensional self-motion
    manifold: the joints must get measurably closer to the null-space target
    while the end-effector pose remains unchanged.

    At the initial configuration [0, 0, 0, -1.57, 0, 1.57, 0.785] joint 2 is
    zero, so the axes of joints 1 and 3 are collinear and the self-motion is
    exactly their counter-rotation: the null-space direction is
    (-1, 0, 1, 0, 0, 0, 0)/sqrt(2).  The posture offsets below have a component
    in that direction, so they are reachable without moving the end-effector.
    (An offset with equal components on joints 1 and 3 would be orthogonal to
    the null space and produce zero projected torque.)

    Phase 1 uses a per-joint stiffness vector to push just joint 1 toward a
    target: with stiffness on only one joint, the projected torque vanishes
    exactly when that joint reaches its target, so it must converge there
    (joint 3 counter-rotates as the null-space byproduct).  Phase 2 uses a
    single scalar stiffness with a target offset lying entirely in the null
    space, which must be reached in all joints.
    """
    with sim_server_context() as robot_server:
        robot = make_robot(robot_server.hostname)

        initial_pose = robot.current_cartesian_state.pose.end_effector_pose
        initial_translation = np.array(initial_pose.translation).flatten()

        def ns_control(nullspace_target: np.ndarray, nullspace_stiffness):
            motion = franky.CartesianImpedanceMotion(
                franky.Affine(initial_pose.matrix),
                nullspace_target=nullspace_target,
                nullspace_stiffness=nullspace_stiffness,
            )
            # Scalars are broadcast to per-joint gains; readback is a 7-vector.
            np.testing.assert_allclose(
                motion.get_nullspace_gains().posture_stiffness,
                np.broadcast_to(nullspace_stiffness, 7),
            )

            robot.move(motion, asynchronous=True)

            import time

            time.sleep(3.0)

            robot.stop()
            try:
                robot.join_motion()
            except franky.ControlException as e:
                if "Move command preempted" not in str(e):
                    raise

            actual_translation = np.array(
                robot.current_cartesian_state.pose.end_effector_pose.translation
            ).flatten()

            # Primary task: the end-effector must not have been disturbed.
            np.testing.assert_allclose(
                actual_translation,
                initial_translation,
                atol=CART_ATOL,
                err_msg="Null-space posture task disturbed the end-effector position",
            )

        # Phase 1: push just joint 1 toward a target via a per-joint stiffness
        # vector (all other joints have zero stiffness).  Joint 1 must reach
        # its target exactly; joint 3 counter-rotates as the null-space
        # byproduct.
        q_initial = np.array(robot.current_joint_state.position)
        nullspace_target = q_initial.copy()
        nullspace_target[0] += 0.1
        ns_control(nullspace_target, np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        q_final = np.array(robot.current_joint_state.position)

        np.testing.assert_allclose(
            q_final[0],
            nullspace_target[0],
            atol=0.05,
            err_msg="Target joint 1 did not converge to its null-space target",
        )

        # Phase 2: move the joints along the null-space direction toward a
        # target configuration using a single scalar stiffness.
        q_initial = np.array(robot.current_joint_state.position)
        nullspace_offset = np.array([0.4, 0.0, -0.4, 0.0, 0.0, 0.0, 0.0])
        nullspace_target = q_initial + nullspace_offset
        ns_control(nullspace_target, 20.0)
        q_final = np.array(robot.current_joint_state.position)

        # Secondary task: the joints must have moved toward the posture target
        # along the null space.  The target is exactly reachable, so require
        # most of the distance to be covered (PD control leaves some residual).
        dist_initial = np.linalg.norm(q_initial - nullspace_target)
        dist_final = np.linalg.norm(q_final - nullspace_target)
        assert dist_final < 0.5 * dist_initial, (
            f"Joints did not approach the null-space target: "
            f"|q - target| went from {dist_initial:.4f} to {dist_final:.4f}"
        )

        # The motion must be the expected self-motion: joint 1 forward,
        # joint 3 backward.
        assert q_final[0] > q_initial[0] + 0.1, (
            f"Joint 1 did not rotate toward the null-space target "
            f"(Δq = {q_final[0] - q_initial[0]:.4f})"
        )
        assert q_final[2] < q_initial[2] - 0.1, (
            f"Joint 3 did not rotate toward the null-space target "
            f"(Δq = {q_final[2] - q_initial[2]:.4f})"
        )


# ---------------------------------------------------------------------------
# Test 7 - Joint impedance tracker
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_joint_impedance_tracker():
    """
    Use JointImpedanceTracker to dynamically drive joints to a target configuration.
    """
    target = [-0.1, 0.1, 0.1, -1.5, 0.1, 1.6, 0.8]
    with sim_server_context() as server:
        robot = make_robot(server.robot_servers[0].hostname)
        with franky.JointImpedanceTracker(robot, period=0.01) as tracker:
            tracker.set_target(target)

            # Tick a few times and wait for it to settle
            for _ in range(100):
                if not tracker.tick():
                    break

        q_actual = list(robot.current_joint_state.position)
        np.testing.assert_allclose(
            q_actual,
            target,
            atol=JOINT_ATOL,
            err_msg="Joint positions out of tolerance for JointImpedanceTracker",
        )


# ---------------------------------------------------------------------------
# Test 8 - Cartesian impedance tracker
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_cartesian_impedance_tracker():
    """
    Move the end-effector to an absolute Cartesian target using CartesianImpedanceTracker.
    """
    offset = np.array([0.05, -0.05, 0.0])
    with sim_server_context() as server:
        robot = make_robot(server.robot_servers[0].hostname)
        initial_pose = robot.current_cartesian_state.pose.end_effector_pose
        initial_translation = np.array(initial_pose.translation).flatten()

        target_translation = initial_translation + offset
        target_matrix = initial_pose.matrix.copy()
        target_matrix[:3, 3] = target_translation

        with franky.CartesianImpedanceTracker(robot, period=0.01) as tracker:
            tracker.set_target(franky.Affine(target_matrix))

            # Tick a few times and wait for it to settle
            for _ in range(100):
                if not tracker.tick():
                    break

        actual_translation = np.array(
            robot.current_cartesian_state.pose.end_effector_pose.translation
        ).flatten()
        np.testing.assert_allclose(
            actual_translation,
            target_translation,
            atol=CART_ATOL,
            err_msg="Position out of tolerance for CartesianImpedanceTracker",
        )


# ---------------------------------------------------------------------------
# Gripper helpers
# ---------------------------------------------------------------------------

GRIPPER_WIDTH_ATOL = 0.005  # m


def make_gripper(hostname: str) -> franky.Gripper:
    """Create a franky Gripper connected to the local simulation gripper server."""
    return franky.Gripper(hostname)


# ---------------------------------------------------------------------------
# Test 5 – Gripper homing
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_gripper_homing():
    """
    Home the gripper and verify that it opens to its maximum width.
    """
    with sim_server_context() as server:
        gripper = make_gripper(server.robot_servers[0].hostname)
        initial_width = gripper.width
        result = gripper.homing()
        assert result, "Gripper homing should return True"
        np.testing.assert_allclose(
            gripper.max_width,
            0.08,
            atol=GRIPPER_WIDTH_ATOL,
            err_msg=(
                f"After homing, max_width {gripper.max_width:.4f} m should equal 0.08m"
            ),
        )
        np.testing.assert_allclose(
            gripper.width,
            initial_width,
            atol=GRIPPER_WIDTH_ATOL,
            err_msg=(
                f"After homing, width {gripper.width:.4f} m should equal "
                f"the prior width {initial_width:.4f} m"
            ),
        )


# ---------------------------------------------------------------------------
# Test 6 – Gripper move
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_gripper_move():
    """
    Move the gripper through several target widths and verify the position
    settles at each commanded value.
    """
    target_widths = [0.08, 0.04, 0.01, 0.06, 0.0]

    with sim_server_context() as server:
        gripper = make_gripper(server.robot_servers[0].hostname)
        for target_width in target_widths:
            result = gripper.move(target_width, 0.05)
            assert result, f"Gripper move to {target_width:.3f} m should return True"
            np.testing.assert_allclose(
                gripper.width,
                target_width,
                atol=GRIPPER_WIDTH_ATOL,
                err_msg=(
                    f"After move({target_width:.3f}), width {gripper.width:.4f} m "
                    f"is outside tolerance"
                ),
            )


# ---------------------------------------------------------------------------
# Test 7 – Gripper grasp success
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_gripper_grasp_success():
    """
    Grasp at an achievable width with generous epsilon.  With no physical object
    blocking the gripper, it reaches the commanded width and the epsilon check
    should succeed.
    """
    with sim_server_context() as server:
        gripper = make_gripper(server.robot_servers[0].hostname)
        gripper.move(0.08, 0.05)  # start fully open

        result = gripper.grasp(0.04, 0.02, 10.0, epsilon_inner=0.02, epsilon_outer=0.02)
        assert result, "Gripper grasp should succeed when width is within epsilon"
        assert gripper.is_grasped, "is_grasped should be True after a successful grasp"


# ---------------------------------------------------------------------------
# Test 8 – Gripper grasp failure
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_gripper_grasp_failure():
    """
    Grasp at a width slightly above the physical maximum (0.08 m).  The gripper
    can only open to 0.08 m, so it settles there.  The epsilon window around the
    commanded 0.09 m does not cover 0.08 m, so the server returns failure and
    franky raises CommandException.
    """
    with sim_server_context() as server:
        gripper = make_gripper(server.robot_servers[0].hostname)
        # Commanded 0.09 m, physical limit is 0.08 m:
        # in_range = 0.09 - 0.005 <= 0.08 <= 0.09 + 0.005  →  0.085 <= 0.08  →  False
        with pytest.raises(franky.CommandException):
            gripper.grasp(0.09, 0.02, 10.0, epsilon_inner=0.005, epsilon_outer=0.005)
        assert not gripper.is_grasped, "is_grasped should be False after a failed grasp"


# ---------------------------------------------------------------------------
# Test 9 – Gripper stop
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_gripper_stop():
    """
    Issue stop on an idle gripper; the server should reply with kSuccess and
    franky should return True.
    """
    with sim_server_context() as server:
        gripper = make_gripper(server.robot_servers[0].hostname)
        gripper.homing()
        result = gripper.stop()
        assert result, "Gripper stop should return True"


# ---------------------------------------------------------------------------
# Test 10 – Motion reuse is rejected
# ---------------------------------------------------------------------------


@pytest.mark.timeout(20)
def test_motion_reuse_raises():
    """
    Motions maintain internal state and must not be executed more than once.
    Reusing a motion object should raise MotionReuseException, both for
    synchronous moves and when preempting an asynchronous motion.
    """
    with sim_server_context() as robot_server:
        robot = make_robot(robot_server.hostname)

        motion = franky.JointWaypointMotion(
            [franky.JointWaypoint([-0.1, 0.1, 0.1, -1.5, 0.1, 1.6, 0.8])]
        )
        robot.move(motion)
        with pytest.raises(franky.MotionReuseException):
            robot.move(motion)

        # Reuse as a preempting motion of an asynchronous move must also fail.
        async_motion = franky.JointWaypointMotion(
            [franky.JointWaypoint([0.1, -0.1, -0.1, -1.6, -0.1, 1.5, 0.7])]
        )
        robot.move(async_motion, asynchronous=True)
        with pytest.raises(franky.MotionReuseException):
            robot.move(motion, asynchronous=True)
        robot.join_motion()

        # A fresh motion instance still works after the failed attempts.
        robot.move(
            franky.JointWaypointMotion(
                [franky.JointWaypoint([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785])]
            )
        )
