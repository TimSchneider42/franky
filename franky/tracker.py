from __future__ import annotations

import time as _time
from typing import Optional

import numpy as np

from ._franky import (
    Affine,
    CartesianImpedanceTrackingMotion,
    CartesianReference,
    ControlException,
    FrictionCompensationParams,
    JointImpedanceGains,
    JointImpedanceTrackingMotion,
    JointReference,
    TorqueStopMotion,
    Twist,
    TwistAcceleration,
)
from .robot import Robot


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
        robot: Robot,
        *,
        translational_stiffness: float = 500.0,
        rotational_stiffness: float = 50.0,
        translational_error_clip: Optional[np.ndarray] = None,
        rotational_error_clip: Optional[np.ndarray] = None,
        nullspace_target: Optional[np.ndarray] = None,
        nullspace_stiffness: float = 0.0,
        friction: Optional[FrictionCompensationParams] = None,
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
        self._period = period
        self._tick_count = 0
        self._t_start = _time.perf_counter()
        self._t_next = self._t_start

        kwargs = {
            "translational_stiffness": translational_stiffness,
            "rotational_stiffness": rotational_stiffness,
            "translational_error_clip": translational_error_clip,
            "rotational_error_clip": rotational_error_clip,
            "nullspace_target": nullspace_target,
            "nullspace_stiffness": nullspace_stiffness,
            "friction": friction,
            "max_delta_tau": max_delta_tau,
            "lower_joint_limits": lower_joint_limits,
            "upper_joint_limits": upper_joint_limits,
            "joint_limit_activation_distance": joint_limit_activation_distance,
            "joint_limit_stiffness": joint_limit_stiffness,
            "joint_limit_damping": joint_limit_damping,
            "joint_limit_max_torque": joint_limit_max_torque,
            "gains_time_constant": gains_time_constant,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        self._motion = CartesianImpedanceTrackingMotion(**kwargs)

        # Seed initial target from current pose so the robot doesn't jump.
        initial_pose = self._robot.current_pose.end_effector_pose
        self._motion.set_reference(CartesianReference(target=initial_pose))

        self._robot.move(self._motion, asynchronous=True)

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

    def set_target(
        self,
        pose: Affine,
        twist: Optional[Twist] = None,
        acceleration: Optional[TwistAcceleration] = None,
    ) -> None:
        """Update the Cartesian target pose and optional twist/acceleration feedforward."""
        kwargs = {"target": pose}
        if twist is not None:
            kwargs["target_twist"] = twist
        if acceleration is not None:
            kwargs["target_acceleration"] = acceleration
        self._motion.set_reference(CartesianReference(**kwargs))

    def set_gains(
        self,
        *,
        translational_stiffness: Optional[float] = None,
        rotational_stiffness: Optional[float] = None,
        nullspace_stiffness: Optional[float] = None,
    ) -> None:
        """Update impedance gains. Smoothed in the RT loop via exponential interpolation.

        Omitted components keep their current target value. A named stiffness overwrites its
        3x3 block isotropically; damping is preserved (use ``motion.set_gains`` for
        anisotropic or explicit-damping gains).
        """
        current = self._motion.get_gains()
        # Preserve the full stiffness matrix (anisotropy) and damping; overwrite only named blocks.
        stiffness = np.array(current.stiffness, copy=True)
        if translational_stiffness is not None:
            stiffness[0:3, 0:3] = translational_stiffness * np.eye(3)
        if rotational_stiffness is not None:
            stiffness[3:6, 3:6] = rotational_stiffness * np.eye(3)
        current.stiffness = stiffness
        self._motion.set_gains(current)
        if nullspace_stiffness is not None:
            nullspace_gains = self._motion.get_nullspace_gains()
            nullspace_gains.posture_stiffness = nullspace_stiffness
            self._motion.set_nullspace_gains(nullspace_gains)

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

    def stop(self, stop_motion: Optional[TorqueStopMotion] = None) -> None:
        """Gracefully stop the tracking controller and wait for the arm to come to rest.

        Enqueues a :class:`TorqueStopMotion` that ramps the last commanded torque
        into a damping-only law, brings the arm to rest, and finishes cleanly
        (no preemption exception). Pass ``stop_motion`` to override the ramp/damping
        behaviour; otherwise sensible defaults are used.

        If the controller is no longer in control (e.g. it already faulted), this
        just joins the motion to surface any stored exception.
        """
        if self._robot.is_in_control:
            self._robot.move(stop_motion or TorqueStopMotion(), asynchronous=True)
        try:
            self._robot.join_motion()
        except ControlException as exc:
            # A graceful TorqueStopMotion finishes without preemption. Tolerate a
            # self-preemption only as a defensive fallback (e.g. if control ended
            # abruptly before we could enqueue the stop).
            if _is_premption_exception(exc):
                return
            raise

    @property
    def motion(self) -> CartesianImpedanceTrackingMotion:
        """The underlying tracking motion instance."""
        return self._motion

    # --- context manager ---

    def __enter__(self) -> CartesianImpedanceTracker:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            self.stop()
        except ControlException:
            # If the body is already unwinding due to another exception
            # (especially KeyboardInterrupt), do not let a cleanup fault mask it.
            if exc_type is not None:
                return False
            raise
        return False


class JointImpedanceTracker:
    """A long-lived session for streaming joint impedance tracking commands.

    Passing ``cartesian_stiffness`` (a 6-vector ``[x, y, z, rx, ry, rz]`` in the base
    frame at the end-effector) enables hybrid Cartesian gain shaping: the controller
    adds ``J^T diag(cartesian_stiffness) J`` on top of the joint-space stiffness each
    cycle (and likewise for ``cartesian_damping``, defaulting to critical damping when
    omitted). The hybrid path is fixed for the lifetime of the motion.

    Example::

        with JointImpedanceTracker(robot, stiffness=[6.0]*7, period=0.01) as tracker:
            while tracker.tick():
                tracker.set_target(q_desired)
    """

    def __init__(
        self,
        robot: Robot,
        *,
        stiffness: Optional[np.ndarray] = None,
        damping: Optional[np.ndarray] = None,
        cartesian_stiffness: Optional[np.ndarray] = None,
        cartesian_damping: Optional[np.ndarray] = None,
        constant_torque_offset: Optional[np.ndarray] = None,
        compensate_coriolis: bool = True,
        friction_coulomb: Optional[np.ndarray] = None,
        friction_viscous: Optional[np.ndarray] = None,
        friction_max_torque: Optional[np.ndarray] = None,
        friction_velocity_epsilon: float = 0.03,
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
        self._period = period
        self._tick_count = 0
        self._t_start = _time.perf_counter()
        self._t_next = self._t_start

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

        kwargs = {
            "stiffness": stiffness_init,
            "damping": damping_init,
            "cartesian_stiffness": (
                np.asarray(cartesian_stiffness, dtype=float)
                if cartesian_stiffness is not None
                else None
            ),
            "cartesian_damping": (
                np.asarray(cartesian_damping, dtype=float)
                if cartesian_damping is not None
                else None
            ),
            "constant_torque_offset": constant_torque_offset,
            "compensate_coriolis": compensate_coriolis,
            "friction_coulomb": friction_coulomb,
            "friction_viscous": friction_viscous,
            "friction_max_torque": friction_max_torque,
            "friction_velocity_epsilon": friction_velocity_epsilon,
            "max_delta_tau": max_delta_tau,
            "lower_joint_limits": lower_joint_limits,
            "upper_joint_limits": upper_joint_limits,
            "joint_limit_activation_distance": joint_limit_activation_distance,
            "joint_limit_stiffness": joint_limit_stiffness,
            "joint_limit_damping": joint_limit_damping,
            "joint_limit_max_torque": joint_limit_max_torque,
            "gains_time_constant": gains_time_constant,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        self._motion = JointImpedanceTrackingMotion(**kwargs)

        # Seed initial target from current joint positions.
        q = self._robot.current_joint_positions
        self._motion.set_reference(JointReference(q=q))

        self._robot.move(self._motion, asynchronous=True)

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
        kwargs = {"q": q}
        if dq is not None:
            kwargs["dq"] = dq
        if tau_ff is not None:
            kwargs["tau_ff"] = tau_ff
        self._motion.set_reference(JointReference(**kwargs))

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
        current = self._motion.get_gains()
        k = _as_joint_gain(
            "stiffness",
            (stiffness if stiffness is not None else current.stiffness),
        )
        d = (
            _as_joint_gain("damping", damping)
            if damping is not None
            else (
                _default_joint_damping(k)
                if stiffness is not None
                else _as_joint_gain("damping", current.damping)
            )
        )
        self._motion.set_gains(JointImpedanceGains(k, d))

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

    def stop(self, stop_motion: Optional[TorqueStopMotion] = None) -> None:
        """Gracefully stop the tracking controller and wait for the arm to come to rest.

        Enqueues a :class:`TorqueStopMotion` that ramps the last commanded torque
        into a damping-only law, brings the arm to rest, and finishes cleanly
        (no preemption exception). Pass ``stop_motion`` to override the ramp/damping
        behaviour; otherwise sensible defaults are used.

        If the controller is no longer in control (e.g. it already faulted), this
        just joins the motion to surface any stored exception.
        """
        if self._robot.is_in_control:
            self._robot.move(stop_motion or TorqueStopMotion(), asynchronous=True)
        try:
            self._robot.join_motion()
        except ControlException as exc:
            # A graceful TorqueStopMotion finishes without preemption. Tolerate a
            # self-preemption only as a defensive fallback (e.g. if control ended
            # abruptly before we could enqueue the stop).
            if _is_premption_exception(exc):
                return
            raise

    @property
    def motion(self) -> JointImpedanceTrackingMotion:
        """The underlying tracking motion instance."""
        return self._motion

    # --- context manager ---

    def __enter__(self) -> JointImpedanceTracker:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            self.stop()
        except ControlException:
            # If the body is already unwinding due to another exception
            # (especially KeyboardInterrupt), do not let a cleanup fault mask it.
            if exc_type is not None:
                return False
            raise
        return False
