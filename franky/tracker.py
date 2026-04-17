from __future__ import annotations

import time as _time
from typing import Optional

import numpy as np

from ._franky import (
    Affine,
    CartesianImpedanceGainsHandle,
    CartesianImpedanceTrackingMotion,
    CartesianReferenceHandle,
    ControlException,
    JointImpedanceGainsHandle,
    JointImpedanceTrackingMotion,
    JointReferenceHandle,
    Twist,
)


def _is_premption_exception(exc: ControlException) -> bool:
    """Return whether this is a preemption error from libfranka."""
    return "Move command preempted!" in str(exc)


_DEFAULT_JOINT_STIFFNESS = np.full(7, 50.0)


def _default_joint_damping(stiffness: np.ndarray) -> np.ndarray:
    return 2.0 * np.sqrt(stiffness)


def _as_joint_gain(name: str, value) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (7,):
        raise ValueError(f"{name} must contain exactly 7 values")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(vector < 0.0):
        raise ValueError(f"{name} must contain only non-negative values")
    return vector.copy()


class CartesianImpedanceTracker:
    """A long-lived session for streaming Cartesian impedance tracking commands.

    Wraps the lifecycle of a CartesianReferenceHandle, CartesianImpedanceTrackingMotion,
    and the async robot.move() call into a single object. Use as a context manager to
    ensure the controller is stopped on exit.

    Example::

        with CartesianImpedanceTracker(robot, translational_stiffness=800.0, period=0.01) as tracker:
            while tracker.tick():
                tracker.set_target(desired_pose)
    """

    def __init__(
        self,
        robot,
        *,
        translational_stiffness: float = 2000.0,
        rotational_stiffness: float = 200.0,
        translational_error_clip: Optional[np.ndarray] = None,
        rotational_error_clip: Optional[np.ndarray] = None,
        nullspace_target: Optional[np.ndarray] = None,
        nullspace_stiffness: float = 0.0,
        max_delta_tau: float = 1.0,
        lower_joint_limits: Optional[np.ndarray] = None,
        upper_joint_limits: Optional[np.ndarray] = None,
        joint_limit_activation_distance: float = 0.1,
        joint_limit_stiffness: float = 4.0,
        joint_limit_damping: float = 1.0,
        joint_limit_max_torque: float = 5.0,
        gains_time_constant: float = 0.1,
        period: Optional[float] = None,
    ):
        self._robot = robot
        self._reference_handle = CartesianReferenceHandle()
        self._gains_handle = CartesianImpedanceGainsHandle()
        self._period = period
        self._tick_count = 0
        self._t_start = _time.perf_counter()
        self._t_next = self._t_start

        # Seed initial target from current pose so the robot doesn't jump.
        initial_pose = self._robot.current_pose.end_effector_pose
        self._reference_handle.set(initial_pose)

        # Seed gains handle so the RT loop has a target from the start.
        self._gains_handle.set(
            translational_stiffness, rotational_stiffness, nullspace_stiffness
        )

        kwargs = {
            "translational_stiffness": translational_stiffness,
            "rotational_stiffness": rotational_stiffness,
            "translational_error_clip": translational_error_clip,
            "rotational_error_clip": rotational_error_clip,
            "nullspace_target": nullspace_target,
            "nullspace_stiffness": nullspace_stiffness,
            "max_delta_tau": max_delta_tau,
            "lower_joint_limits": lower_joint_limits,
            "upper_joint_limits": upper_joint_limits,
            "joint_limit_activation_distance": joint_limit_activation_distance,
            "joint_limit_stiffness": joint_limit_stiffness,
            "joint_limit_damping": joint_limit_damping,
            "joint_limit_max_torque": joint_limit_max_torque,
            "gains_handle": self._gains_handle,
            "gains_time_constant": gains_time_constant,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        motion = CartesianImpedanceTrackingMotion(
            reference_handle=self._reference_handle, **kwargs
        )
        self._robot.move(motion, asynchronous=True)

    # --- tick ---

    def tick(self) -> bool:
        """Sleep to maintain the requested period and return whether the controller is alive.

        On the first call, returns immediately (no sleep). On subsequent calls,
        sleeps the remaining time until the next tick boundary so that loop body
        time is compensated for.

        If no period was set, just returns is_running without sleeping.
        """
        if self._period is not None and self._tick_count > 0:
            now = _time.perf_counter()
            remaining = self._t_next - now
            if remaining > 0:
                _time.sleep(remaining)
            self._t_next += self._period
        elif self._period is not None:
            # First tick: set up the schedule.
            self._t_next = _time.perf_counter() + self._period

        if not self._robot.is_in_control:
            return False

        self._tick_count += 1
        return True

    # --- streaming updates ---

    def set_target(self, pose: Affine, twist: Optional[Twist] = None) -> None:
        """Update the Cartesian target pose and optional twist (feedforward velocity)."""
        if twist is not None:
            self._reference_handle.set(pose, twist)
        else:
            self._reference_handle.set(pose)

    def set_gains(
        self,
        *,
        translational_stiffness: Optional[float] = None,
        rotational_stiffness: Optional[float] = None,
        nullspace_stiffness: Optional[float] = None,
    ) -> None:
        """Update impedance gains. Smoothed in the RT loop via exponential interpolation.

        Only the provided gains are changed; omitted gains keep their current target values.
        """
        current = self._gains_handle.get() if self._gains_handle.has_gains else None
        ts = (
            translational_stiffness
            if translational_stiffness is not None
            else (current.translational_stiffness if current else 2000.0)
        )
        rs = (
            rotational_stiffness
            if rotational_stiffness is not None
            else (current.rotational_stiffness if current else 200.0)
        )
        ns = (
            nullspace_stiffness
            if nullspace_stiffness is not None
            else (current.nullspace_stiffness if current else 0.0)
        )
        self._gains_handle.set(ts, rs, ns)

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

    @property
    def elapsed_time(self) -> float:
        """Seconds since the tracker was created."""
        return _time.perf_counter() - self._t_start

    @property
    def tick_count(self) -> int:
        """Number of ticks that have returned True."""
        return self._tick_count

    # --- lifecycle ---

    def stop(self) -> None:
        """Stop the tracking controller and wait for the motion to finish."""
        requested_stop = self._robot.is_in_control
        if self._robot.is_in_control:
            self._robot.stop()
        try:
            self._robot.join_motion()
        except ControlException as exc:
            # libfranka reports our own stop() as a preempted move when the
            # asynchronous control thread unwinds. Treat that as normal shutdown.
            if requested_stop and _is_premption_exception(exc):
                return
            raise

    # --- escape hatches ---

    @property
    def reference_handle(self) -> CartesianReferenceHandle:
        """The underlying reference handle, for direct access from tight loops or C++ interop."""
        return self._reference_handle

    @property
    def gains_handle(self) -> CartesianImpedanceGainsHandle:
        """The underlying gains handle, for direct access from tight loops or C++ interop."""
        return self._gains_handle

    # --- context manager ---

    def __enter__(self) -> CartesianImpedanceTracker:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            self.stop()
        except ControlException as stop_exc:
            # If the body is already unwinding due to another exception
            # (especially KeyboardInterrupt), do not let cleanup mask it.
            # Only ignore the specific self-preemption raised by libfranka in
            # response to our stop() call; propagate real controller faults.
            if exc_type is not None and _is_premption_exception(stop_exc):
                return False
            raise
        return False


class JointImpedanceTracker:
    """A long-lived session for streaming joint impedance tracking commands.

    Example::

        with JointImpedanceTracker(robot, stiffness=[6.0]*7, period=0.01) as tracker:
            while tracker.tick():
                tracker.set_target(q_desired)
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
        gains_time_constant: float = 0.1,
        period: Optional[float] = None,
    ):
        self._robot = robot
        self._reference_handle = JointReferenceHandle()
        self._gains_handle = JointImpedanceGainsHandle()
        self._period = period
        self._tick_count = 0
        self._t_start = _time.perf_counter()
        self._t_next = self._t_start

        # Seed initial target from current joint positions.
        q = self._robot.current_joint_positions
        self._reference_handle.set(q)

        # Seed gains handle with initial values. When damping is omitted, use
        # critical damping for unit inertia so zero stiffness implies zero damping.
        stiffness_init = _as_joint_gain(
            "stiffness", _DEFAULT_JOINT_STIFFNESS if stiffness is None else stiffness
        )
        damping_init = (
            _as_joint_gain("damping", damping)
            if damping is not None
            else _default_joint_damping(stiffness_init)
        )
        self._gains_handle.set(stiffness_init, damping_init)

        kwargs = {
            "stiffness": stiffness_init,
            "damping": damping_init,
            "constant_torque_offset": constant_torque_offset,
            "compensate_coriolis": compensate_coriolis,
            "max_delta_tau": max_delta_tau,
            "lower_joint_limits": lower_joint_limits,
            "upper_joint_limits": upper_joint_limits,
            "joint_limit_activation_distance": joint_limit_activation_distance,
            "joint_limit_stiffness": joint_limit_stiffness,
            "joint_limit_damping": joint_limit_damping,
            "joint_limit_max_torque": joint_limit_max_torque,
            "gains_handle": self._gains_handle,
            "gains_time_constant": gains_time_constant,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        motion = JointImpedanceTrackingMotion(
            reference_handle=self._reference_handle, **kwargs
        )
        self._robot.move(motion, asynchronous=True)

    # --- tick ---

    def tick(self) -> bool:
        """Sleep to maintain the requested period and return whether the controller is alive.

        On the first call, returns immediately (no sleep). On subsequent calls,
        sleeps the remaining time until the next tick boundary so that loop body
        time is compensated for.

        If no period was set, just returns is_running without sleeping.
        """
        if self._period is not None and self._tick_count > 0:
            now = _time.perf_counter()
            remaining = self._t_next - now
            if remaining > 0:
                _time.sleep(remaining)
            self._t_next += self._period
        elif self._period is not None:
            self._t_next = _time.perf_counter() + self._period

        if not self._robot.is_in_control:
            return False

        self._tick_count += 1
        return True

    # --- streaming updates ---

    def set_target(
        self,
        q: np.ndarray,
        dq: Optional[np.ndarray] = None,
        tau_ff: Optional[np.ndarray] = None,
    ) -> None:
        """Update the joint target position, optional velocity, and optional feedforward torque."""
        self._reference_handle.set(q, dq, tau_ff)

    def set_gains(
        self,
        *,
        stiffness: Optional[np.ndarray] = None,
        damping: Optional[np.ndarray] = None,
    ) -> None:
        """Update joint impedance gains. Smoothed in the RT loop via exponential interpolation.

        When stiffness is changed and damping is omitted, damping is updated to
        the critical damping heuristic 2*sqrt(stiffness). Otherwise, omitted
        gains keep their current target values.
        """
        current = self._gains_handle.get() if self._gains_handle.has_gains else None
        k = _as_joint_gain(
            "stiffness",
            (
                stiffness
                if stiffness is not None
                else (current.stiffness if current else _DEFAULT_JOINT_STIFFNESS)
            ),
        )
        d = (
            _as_joint_gain("damping", damping)
            if damping is not None
            else (
                _default_joint_damping(k)
                if stiffness is not None or current is None
                else _as_joint_gain("damping", current.damping)
            )
        )
        self._gains_handle.set(k, d)

    # --- state ---

    @property
    def state(self):
        return self._robot.state

    @property
    def is_running(self) -> bool:
        return self._robot.is_in_control

    @property
    def elapsed_time(self) -> float:
        """Seconds since the tracker was created."""
        return _time.perf_counter() - self._t_start

    @property
    def tick_count(self) -> int:
        """Number of ticks that have returned True."""
        return self._tick_count

    # --- lifecycle ---

    def stop(self) -> None:
        requested_stop = self._robot.is_in_control
        if self._robot.is_in_control:
            self._robot.stop()
        try:
            self._robot.join_motion()
        except ControlException as exc:
            # libfranka reports our own stop() as a preempted move when the
            # asynchronous control thread unwinds. Treat that as normal shutdown.
            if requested_stop and _is_premption_exception(exc):
                return
            raise

    # --- escape hatches ---

    @property
    def reference_handle(self) -> JointReferenceHandle:
        return self._reference_handle

    @property
    def gains_handle(self) -> JointImpedanceGainsHandle:
        return self._gains_handle

    # --- context manager ---

    def __enter__(self) -> JointImpedanceTracker:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            self.stop()
        except ControlException as stop_exc:
            # Only ignore the specific preemption raised by libfranka in
            # response to our stop() call; propagate real controller faults.
            if exc_type is not None and _is_premption_exception(stop_exc):
                return False
            raise
        return False
