#include "franky/motion/joint_impedance_tracking_motion.hpp"

namespace franky {

JointImpedanceTrackingMotion::JointImpedanceTrackingMotion(const Params &params, double gains_time_constant)
    : JointImpedanceBase(Vector7d::Zero(), Vector7d::Zero(), params, gains_time_constant) {}

JointImpedanceTrackingMotion::JointImpedanceTrackingMotion(
    ReferenceCallback reference_callback, const Params &params, double gains_time_constant)
    : JointImpedanceBase(Vector7d::Zero(), Vector7d::Zero(), params, gains_time_constant),
      reference_callback_(std::move(reference_callback)) {}

void JointImpedanceTrackingMotion::initImpl(
    const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) {
  target_ = robot_state.q;
  target_velocity_.setZero();
}

franka::Torques JointImpedanceTrackingMotion::nextCommandImpl(
    const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
    const std::optional<franka::Torques> &previous_command) {
  JointReference reference;
  reference.q = target_;
  reference.dq = target_velocity_;

  auto opt_reference = reference_handle_.getUnsafe();
  if (opt_reference) {
    reference = *opt_reference;
  } else if (reference_callback_) {
    reference = reference_callback_(robot_state, time_step, rel_time, abs_time);
  }

  target_ = reference.q;
  target_velocity_ = reference.dq;
  const double dt = time_step.toSec();
  return computeCommand(robot_state, reference, dt);
}

}  // namespace franky
