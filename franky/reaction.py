from ._franky import (
    Condition,
    BaseCartesianPoseMotion,
    BaseCartesianVelocityMotion,
    BaseJointPositionMotion,
    BaseJointVelocityMotion,
    BaseTorqueMotion,
    CartesianPoseReaction as _CartesianPoseReaction,
    CartesianVelocityReaction as _CartesianVelocityReaction,
    JointPositionReaction as _JointPositionReaction,
    JointVelocityReaction as _JointVelocityReaction,
    TorqueReaction as _TorqueReaction,
)

from .motion import Motion


class Reaction:
    """A reaction that can be attached to a motion.

    Reactions consist of a condition and a motion that replaces the current motion immediately if
    the condition is met. This class dispatches to the reaction type matching the given motion
    (e.g. :class:`CartesianPoseReaction` for Cartesian pose motions), as reactions can only
    replace motions of the same control signal type.

    Args:
        condition: The condition that must be met for the reaction to be executed.
        motion: The motion that replaces the current motion if the condition is met.
    """

    _control_signal_type = None

    def __new__(cls, condition: Condition, motion: Motion):
        for reaction_type in _REACTION_TYPES:
            if isinstance(motion, reaction_type._motion_type):
                return reaction_type.__new__(reaction_type, condition, motion)
        raise TypeError(f"Unknown motion type {type(motion)}.")


class CartesianPoseReaction(_CartesianPoseReaction, Reaction):
    """A reaction that replaces the current motion with a Cartesian pose motion."""

    _motion_type = BaseCartesianPoseMotion


class CartesianVelocityReaction(_CartesianVelocityReaction, Reaction):
    """A reaction that replaces the current motion with a Cartesian velocity motion."""

    _motion_type = BaseCartesianVelocityMotion


class JointPositionReaction(_JointPositionReaction, Reaction):
    """A reaction that replaces the current motion with a joint position motion."""

    _motion_type = BaseJointPositionMotion


class JointVelocityReaction(_JointVelocityReaction, Reaction):
    """A reaction that replaces the current motion with a joint velocity motion."""

    _motion_type = BaseJointVelocityMotion


class TorqueReaction(_TorqueReaction, Reaction):
    """A reaction that replaces the current motion with a torque motion."""

    _motion_type = BaseTorqueMotion


_REACTION_TYPES = [
    CartesianPoseReaction,
    CartesianVelocityReaction,
    JointPositionReaction,
    JointVelocityReaction,
    TorqueReaction,
]
