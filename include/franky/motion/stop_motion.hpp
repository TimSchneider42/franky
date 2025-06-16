#pragma once

#include "franky/motion/cartesian_motion.hpp"
#include "franky/motion/cartesian_velocity_motion.hpp"
#include "franky/motion/joint_motion.hpp"
#include "franky/motion/joint_velocity_motion.hpp"
#include "franky/motion/reference_type.hpp"

namespace franky {

template <typename ControlSignalType>
class StopMotion;

/**
 * @brief Stop motion for joint position control mode.
 */
template <>
class StopMotion<franka::JointPositions> : public JointMotion {
 public:
  /**
   *
   * @param relative_dynamics_factor Relative dynamics factor for this stop
   * motion.
   */
  explicit StopMotion(const RelativeDynamicsFactor &relative_dynamics_factor = 1.0)
      : JointMotion(JointState(Vector7d::Zero()), ReferenceType::kRelative, relative_dynamics_factor) {}
};

/**
 * @brief Stop motion for joint velocity position control mode.
 */
template <>
class StopMotion<franka::JointVelocities> : public JointVelocityMotion {
 public:
  /**
   *
   * @param relative_dynamics_factor Relative dynamics factor for this stop
   * motion.
   */
  explicit StopMotion(const RelativeDynamicsFactor &relative_dynamics_factor = 1.0)
      : JointVelocityMotion(Vector7d::Zero(), franka::Duration(0), relative_dynamics_factor) {}
};

/**
 * @brief Stop motion for cartesian pose control mode.
 */
template <>
class StopMotion<franka::CartesianPose> : public CartesianWaypointMotion {
 public:
  /**
   *
   * @param relative_dynamics_factor Relative dynamics factor for this stop
   * motion.
   */
  explicit StopMotion(const RelativeDynamicsFactor &relative_dynamics_factor = 1.0)
      : CartesianWaypointMotion(
            {PositionWaypoint<CartesianState>{
                {
                    .target = RobotPose(),
                    .hold_target_duration = franka::Duration(50),
                },
                ReferenceType::kRelative}},
            relative_dynamics_factor) {}
};

/**
 * @brief Stop motion for cartesian velocity control mode.
 */
template <>
class StopMotion<franka::CartesianVelocities> : public CartesianVelocityMotion {
 public:
  /**
   *
   * @param relative_dynamics_factor Relative dynamics factor for this stop
   * motion.
   */
  explicit StopMotion(const RelativeDynamicsFactor &relative_dynamics_factor = 1.0)
      : CartesianVelocityMotion(RobotVelocity(), franka::Duration(0), relative_dynamics_factor) {}
};

}  // namespace franky
