#pragma once

#include "franky/motion/reference_type.hpp"
#include "franky/motion/joint_motion.hpp"
#include "franky/motion/cartesian_motion.hpp"

namespace franky {

template<typename ControlSignalType>
class StopMotion;

/**
 * @brief Stop motion for joint position control mode.
 */
template<>
class StopMotion<franka::JointPositions> : public JointMotion {
 public:
  explicit StopMotion() : JointMotion(
      JointState(Vector7d::Zero()),
      ReferenceType::Relative,
      RelativeDynamicsFactor::MAX_DYNAMICS()
  ) {}
};

/**
 * @brief Stop motion for cartesian pose control mode.
 */
template<>
class StopMotion<franka::CartesianPose> : public CartesianMotion {
 public:
  explicit StopMotion() : CartesianMotion(
      RobotPose(Affine::Identity()),
      ReferenceType::Relative,
      RelativeDynamicsFactor::MAX_DYNAMICS()
  ) {}
};

}  // namespace franky