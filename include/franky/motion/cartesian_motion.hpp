#pragma once

#include "franky/motion/cartesian_waypoint_motion.hpp"
#include "franky/motion/reference_type.hpp"

namespace franky {

/**
 * @brief Cartesian motion with a single target.
 */
class CartesianMotion : public CartesianWaypointMotion {
 public:
  /**
   * @brief Construct a Cartesian motion.
   *
   * @param target                   The target Cartesian state.
   * @param reference_type           The reference type (absolute or relative).
   * An absolute target is defined in the robot's base frame, a relative target
   * is defined in the current end-effector frame.
   * @param relative_dynamics_factor The relative dynamics factor for this
   * motion. The factor will get multiplied with the robot's global dynamics
   * factor to get the actual dynamics factor for this motion.
   * @param return_when_finished     Whether to end the motion when the target
   * is reached or keep holding the last target.
   * @param frame                    The end-effector frame for which the target
   * is defined. This is a transformation from the configured end-effector frame
   * to the end-effector frame the target is defined for.
   */
  explicit CartesianMotion(
      const CartesianState &target, ReferenceType reference_type = ReferenceType::kAbsolute,
      const RelativeDynamicsFactor &relative_dynamics_factor = 1.0, bool return_when_finished = true,
      const Affine &frame = Affine::Identity());
};

}  // namespace franky
