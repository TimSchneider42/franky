#pragma once

#include "franky/motion/cartesian_velocity_waypoint_motion.hpp"

namespace franky {

/**
 * @brief Cartesian velocity motion with a single target velocity.
 */
class CartesianVelocityMotion : public CartesianVelocityWaypointMotion {
 public:
  /**
   * @brief Construct a Cartesian motion.
   *
   * @param target                   The target Cartesian velocity.
   * @param max_total_duration       Maximum time this command is active. Default is infinite.
   * @param relative_dynamics_factor The relative dynamics factor for this motion. The factor will get multiplied with
   *                                 the robot's global dynamics factor to get the actual dynamics factor for this
   *                                 motion.
   * @param frame                    The end-effector frame for which the target is defined. This is a transformation
   *                                 from the configured end-effector frame to the end-effector frame the target is
   *                                 defined for.
   */
  explicit CartesianVelocityMotion(
      const RobotVelocity &target,
      const std::optional<franka::Duration> &max_total_duration = std::nullopt,
      const RelativeDynamicsFactor &relative_dynamics_factor = 1.0,
      const Affine &frame = Affine::Identity());
};

}  // namespace franky
