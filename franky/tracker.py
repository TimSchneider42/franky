from __future__ import annotations

from typing import Optional

import numpy as np

from ._franky import (
    Affine,
    CartesianImpedanceTrackingMotion,
    CartesianReferenceHandle,
    JointImpedanceTrackingMotion,
    JointReferenceHandle,
    Twist,
)


class CartesianImpedanceTracker:
    """A long-lived session for streaming Cartesian impedance tracking commands.

    Wraps the lifecycle of a CartesianReferenceHandle, CartesianImpedanceTrackingMotion,
    and the async robot.move() call into a single object. Use as a context manager to
    ensure the controller is stopped on exit.

    Example::

        with CartesianImpedanceTracker(robot, translational_stiffness=800.0) as tracker:
            while tracker.is_running:
                tracker.set_target(desired_pose)
                time.sleep(0.01)
    """

    def __init__(
        self,
        robot,
        *,
        translational_stiffness: float = 2000.0,
        rotational_stiffness: float = 200.0,
        nullspace_target: Optional[np.ndarray] = None,
        nullspace_stiffness: float = 0.0,
        max_delta_tau: float = 1.0,
        lower_joint_limits: Optional[np.ndarray] = None,
        upper_joint_limits: Optional[np.ndarray] = None,
        joint_limit_activation_distance: float = 0.1,
        joint_limit_stiffness: float = 4.0,
        joint_limit_damping: float = 1.0,
        joint_limit_max_torque: float = 5.0,
    ):
        self._robot = robot
        self._reference_handle = CartesianReferenceHandle()

        # Seed initial target from current pose so the robot doesn't jump.
        initial_pose = self._robot.current_pose.end_effector_pose
        self._reference_handle.set(initial_pose)

        kwargs = {
            "translational_stiffness": translational_stiffness,
            "rotational_stiffness": rotational_stiffness,
            "nullspace_target": nullspace_target,
            "nullspace_stiffness": nullspace_stiffness,
            "max_delta_tau": max_delta_tau,
            "lower_joint_limits": lower_joint_limits,
            "upper_joint_limits": upper_joint_limits,
            "joint_limit_activation_distance": joint_limit_activation_distance,
            "joint_limit_stiffness": joint_limit_stiffness,
            "joint_limit_damping": joint_limit_damping,
            "joint_limit_max_torque": joint_limit_max_torque,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        motion = CartesianImpedanceTrackingMotion(
            reference_handle=self._reference_handle, **kwargs
        )
        self._robot.move(motion, asynchronous=True)

    def set_target(self, pose: Affine, twist: Optional[Twist] = None) -> None:
        """Update the Cartesian target pose and optional twist (feedforward velocity)."""
        if twist is not None:
            self._reference_handle.set(pose, twist)
        else:
            self._reference_handle.set(pose)

    # --- state ---

    @property
    def state(self):
        """The current robot state from this control session."""
        return self._robot.state

    @property
    def current_pose(self):
        """The current end-effector pose as a RobotPose (shorthand for robot.current_pose)."""
        return self._robot.current_pose

    @property
    def is_running(self) -> bool:
        """Whether the tracking controller is still active."""
        return self._robot.is_in_control

    # --- lifecycle ---

    def stop(self) -> None:
        """Stop the tracking controller and wait for the motion to finish."""
        if self._robot.is_in_control:
            self._robot.stop()
        self._robot.join_motion()

    # --- escape hatch ---

    @property
    def reference_handle(self) -> CartesianReferenceHandle:
        """The underlying reference handle, for direct access from tight loops or C++ interop."""
        return self._reference_handle

    # --- context manager ---

    def __enter__(self) -> CartesianImpedanceTracker:
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


class JointImpedanceTracker:
    """A long-lived session for streaming joint impedance tracking commands.

    Example::

        with JointImpedanceTracker(robot, stiffness=[6.0]*7) as tracker:
            while tracker.is_running:
                tracker.set_target(q_desired)
                time.sleep(0.01)
    """

    def __init__(
        self,
        robot,
        *,
        stiffness: Optional[np.ndarray] = None,
        damping: Optional[np.ndarray] = None,
        constant_torque_offset: Optional[np.ndarray] = None,
        compensate_coriolis: bool = True,
        max_delta_tau: float = 1.0,
        lower_joint_limits: Optional[np.ndarray] = None,
        upper_joint_limits: Optional[np.ndarray] = None,
        joint_limit_activation_distance: float = 0.1,
        joint_limit_stiffness: float = 4.0,
        joint_limit_damping: float = 1.0,
        joint_limit_max_torque: float = 5.0,
    ):
        self._robot = robot
        self._reference_handle = JointReferenceHandle()

        # Seed initial target from current joint positions.
        q = self._robot.current_joint_positions
        self._reference_handle.set(q)

        kwargs = {
            "stiffness": stiffness,
            "damping": damping,
            "constant_torque_offset": constant_torque_offset,
            "compensate_coriolis": compensate_coriolis,
            "max_delta_tau": max_delta_tau,
            "lower_joint_limits": lower_joint_limits,
            "upper_joint_limits": upper_joint_limits,
            "joint_limit_activation_distance": joint_limit_activation_distance,
            "joint_limit_stiffness": joint_limit_stiffness,
            "joint_limit_damping": joint_limit_damping,
            "joint_limit_max_torque": joint_limit_max_torque,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        motion = JointImpedanceTrackingMotion(
            reference_handle=self._reference_handle, **kwargs
        )
        self._robot.move(motion, asynchronous=True)

    def set_target(
        self,
        q: np.ndarray,
        dq: Optional[np.ndarray] = None,
        tau_ff: Optional[np.ndarray] = None,
    ) -> None:
        """Update the joint target position, optional velocity, and optional feedforward torque."""
        self._reference_handle.set(q, dq, tau_ff)

    # --- state ---

    @property
    def state(self):
        return self._robot.state

    @property
    def is_running(self) -> bool:
        return self._robot.is_in_control

    # --- lifecycle ---

    def stop(self) -> None:
        if self._robot.is_in_control:
            self._robot.stop()
        self._robot.join_motion()

    # --- escape hatch ---

    @property
    def reference_handle(self) -> JointReferenceHandle:
        return self._reference_handle

    # --- context manager ---

    def __enter__(self) -> JointImpedanceTracker:
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
