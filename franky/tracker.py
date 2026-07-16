from __future__ import annotations

import time as _time
from typing import Optional, Union

import numpy as np

from ._franky import (
    Affine,
    CartesianImpedanceGains,
    CartesianImpedanceTrackingMotion,
    CartesianReference,
    ControlException,
    FrictionCompensationParams,
    JointImpedanceGains,
    JointImpedanceTrackingMotion,
    JointReference,
    ManipulabilityTask,
    NullspaceGains,
    PostureTask,
    TorqueStopMotion,
    Twist,
    TwistAcceleration,
)
from .robot import Robot


def _is_premption_exception(exc: ControlException) -> bool:
    """Return whether this is a preemption error from libfranka."""
    return "Move command preempted!" in str(exc)


_DEFAULT_JOINT_STIFFNESS = np.full(7, 50.0)


class _CriticalDamping:
    """Sentinel type for :data:`CRITICAL`."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "franky.CRITICAL"


#: Pass as ``damping=`` to unpin damping back to critical tracking without touching stiffness.
CRITICAL = _CriticalDamping()


def _as_gain_vector(name: str, value, size: int) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (size,):
        raise ValueError(f"{name} must contain exactly {size} values")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(vector < 0.0):
        raise ValueError(f"{name} must contain only non-negative values")
    return vector.copy()


class CartesianImpedanceTracker:
    """Long-lived session for streaming Cartesian impedance tracking commands.

    Use as a context manager to stop the controller on exit. Stiffness and damping are
    orthogonal knobs (see :meth:`set_gains`); omitting damping means critical damping
    (``2 * sqrt(stiffness)``), tracked against the current stiffness each cycle. Pass
    ``posture_task``/``manipulability_task`` to add nullspace objectives.

    Example::

        with CartesianImpedanceTracker(robot, translational_stiffness=800.0, period=0.01) as tracker:
            while tracker.tick():
                tracker.set_target(desired_pose)
    """

    def __init__(
        self,
        robot: Robot,
        *,
        translational_stiffness: Optional[float] = None,
        rotational_stiffness: Optional[float] = None,
        damping: Optional[np.ndarray] = None,
        gains: Optional[CartesianImpedanceGains] = None,
        translational_error_clip: Optional[np.ndarray] = None,
        rotational_error_clip: Optional[np.ndarray] = None,
        posture_task: Optional[PostureTask] = None,
        manipulability_task: Optional[ManipulabilityTask] = None,
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
        if gains is not None and (
            translational_stiffness is not None
            or rotational_stiffness is not None
            or damping is not None
        ):
            raise ValueError(
                "Pass either `gains` (a full CartesianImpedanceGains object, for anisotropic stiffness) "
                "or the isotropic `translational_stiffness`/`rotational_stiffness`/`damping` keywords, "
                "not both."
            )

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
            "posture_task": posture_task,
            "manipulability_task": manipulability_task,
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

        if gains is not None:
            self._motion.set_gains(gains)
        elif damping is not None and damping is not CRITICAL:
            current = self._motion.get_gains()
            current.damping = np.diag(_as_gain_vector("damping", damping, 6))
            self._motion.set_gains(current)

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
        damping: Optional[np.ndarray] = None,
        gains: Optional[CartesianImpedanceGains] = None,
        posture_stiffness: Optional[Union[float, np.ndarray]] = None,
        nullspace_gains: Optional[NullspaceGains] = None,
    ) -> None:
        """Update impedance gains (smoothed in the RT loop).

        ``translational_stiffness``/``rotational_stiffness`` set a stiffness block. ``damping``
        is all-or-nothing: a full 6-vector ``[x, y, z, rx, ry, rz]`` pins it, :data:`CRITICAL`
        unpins it. ``gains`` total-replaces with a full :class:`CartesianImpedanceGains` object
        (for anisotropic stiffness) and is exclusive with the rest. Omitting damping means
        critical: a stiffness change re-criticals unless ``damping`` is passed too.

        For the nullspace, ``posture_stiffness`` (scalar or 7-vector) nudges just the posture
        task's stiffness; ``nullspace_gains`` replaces the full :class:`NullspaceGains` (posture +
        manipulability). Mutually exclusive; both only retune tasks configured at construction.
        """
        stiffness_given = (
            translational_stiffness is not None or rotational_stiffness is not None
        )
        damping_given = damping is not None
        if gains is not None and (stiffness_given or damping_given):
            raise ValueError(
                "Pass either `gains` (a full CartesianImpedanceGains object, for anisotropic stiffness) "
                "or the isotropic `translational_stiffness`/`rotational_stiffness`/`damping` keywords, "
                "not both."
            )
        if nullspace_gains is not None and posture_stiffness is not None:
            raise ValueError(
                "Pass either `nullspace_gains` or `posture_stiffness`, not both."
            )

        if gains is not None:
            self._motion.set_gains(gains)
        elif stiffness_given or damping_given:
            current = self._motion.get_gains()
            # Preserve the full stiffness matrix (anisotropy); overwrite only named blocks.
            stiffness_matrix = np.array(current.stiffness, copy=True)
            if translational_stiffness is not None:
                stiffness_matrix[0:3, 0:3] = translational_stiffness * np.eye(3)
            if rotational_stiffness is not None:
                stiffness_matrix[3:6, 3:6] = rotational_stiffness * np.eye(3)
            current.stiffness = stiffness_matrix

            if damping_given:
                current.damping = (
                    None
                    if damping is CRITICAL
                    else np.diag(_as_gain_vector("damping", damping, 6))
                )
            elif stiffness_given:
                current.damping = None
            self._motion.set_gains(current)

        if nullspace_gains is not None:
            self._motion.set_nullspace_gains(nullspace_gains)
        elif posture_stiffness is not None:
            current_ns = self._motion.get_nullspace_gains()
            current_ns.posture_stiffness = posture_stiffness
            self._motion.set_nullspace_gains(current_ns)

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

        # Damping is all-or-nothing: omitted/CRITICAL -> unset (RT loop tracks critical);
        # a full 7-vector pins it.
        stiffness_init = _as_gain_vector(
            "stiffness", _DEFAULT_JOINT_STIFFNESS if stiffness is None else stiffness, 7
        )
        damping_init = (
            _as_gain_vector("damping", damping, 7)
            if damping is not None and damping is not CRITICAL
            else None
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
        """Update joint impedance gains (smoothed in the RT loop).

        ``stiffness``/``damping`` are orthogonal 7-vectors. ``damping`` is all-or-nothing:
        passing it pins it, :data:`CRITICAL` unpins it. Omitting damping means critical: a
        stiffness change re-criticals unless ``damping`` is passed too.
        """
        if stiffness is None and damping is None:
            return

        current = self._motion.get_gains()
        k = _as_gain_vector(
            "stiffness", (stiffness if stiffness is not None else current.stiffness), 7
        )
        if damping is not None:
            d = None if damping is CRITICAL else _as_gain_vector("damping", damping, 7)
        else:
            d = None
        self._motion.set_gains(JointImpedanceGains(k, d))

    # --- state ---

    @property
    def state(self):
        """The current robot state from this control session."""
        return self._robot.state

    @property
    def current_joint_state(self):
        """The current joint state as a JointState (shorthand for robot.current_joint_state)."""
        return self._robot.current_joint_state

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
