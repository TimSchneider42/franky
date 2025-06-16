from typing import Union

from ._franky import (
    BaseCartesianPoseMotion,
    BaseCartesianVelocityMotion,
    BaseJointPositionMotion,
    BaseJointVelocityMotion,
    BaseTorqueMotion,
)

Motion = Union[
    BaseCartesianPoseMotion,
    BaseCartesianVelocityMotion,
    BaseJointPositionMotion,
    BaseJointVelocityMotion,
    BaseTorqueMotion,
]
