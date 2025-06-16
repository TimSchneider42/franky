#pragma once

namespace franky {

/**
 * @brief Type of control signal.
 */
enum ControlSignalType { Torques, JointVelocities, JointPositions, CartesianVelocities, CartesianPose };

}  // namespace franky
