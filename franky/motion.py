from typing import Union

from ._franky import (
    BaseCartesianPoseMotion,
    BaseCartesianVelocityMotion,
    BaseJointPositionMotion,
    BaseJointVelocityMotion,
    BaseTorqueMotion,
)

#: Union of the base motion types of all control signal types (Cartesian pose, Cartesian
#: velocity, joint position, joint velocity, and torque control). Every motion franky provides is
#: an instance of one of these base types.
Motion = Union[
    BaseCartesianPoseMotion,
    BaseCartesianVelocityMotion,
    BaseJointPositionMotion,
    BaseJointVelocityMotion,
    BaseTorqueMotion,
]
