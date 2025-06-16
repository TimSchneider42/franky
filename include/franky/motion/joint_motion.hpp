#pragma once

#include "franky/joint_state.hpp"
#include "franky/motion/joint_waypoint_motion.hpp"
#include "franky/motion/reference_type.hpp"

namespace franky {

/**
 * @brief Joint motion with a single target.
 */
class JointMotion : public JointWaypointMotion {
 public:
  /**
   * @param target                   The target joint state.
   * @param relative_dynamics_factor The relative dynamics factor for this
   * motion. The factor will get multiplied with the robot's global dynamics
   * factor to get the actual dynamics factor for this motion.
   * @param return_when_finished     Whether to end the motion when the target
   * is reached or keep holding the last target.
   */
  explicit JointMotion(
      const JointState &target, ReferenceType reference_type = ReferenceType::kAbsolute,
      const RelativeDynamicsFactor &relative_dynamics_factor = 1.0, bool return_when_finished = true);
};

}  // namespace franky
