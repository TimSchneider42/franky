#pragma once

#include <optional>

#include "franky/motion/joint_impedance_base.hpp"
#include "franky/types.hpp"

namespace franky {

/**
 * @brief Client-side joint impedance controller.
 *
 * This motion uses Franka's torque interface and regulates toward a constant
 * joint target (and optional target velocity). It does not terminate on its
 * own; it runs until it is preempted or stopped.
 */
class JointImpedanceMotion : public JointImpedanceBase {
 public:
  using Params = JointImpedanceParams;

  explicit JointImpedanceMotion(const Vector7d &target);
  explicit JointImpedanceMotion(const Vector7d &target, const Params &params);
  JointImpedanceMotion(const Vector7d &target, const Vector7d &target_velocity);
  JointImpedanceMotion(const Vector7d &target, const Vector7d &target_velocity, const Params &params);

 protected:
  franka::Torques nextCommandImpl(
      const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
      const std::optional<franka::Torques> &previous_command) override;
};

}  // namespace franky
