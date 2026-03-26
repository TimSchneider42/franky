#include "franky/motion/joint_impedance_tracking_motion.hpp"

namespace franky {

void JointReferenceHandle::set(const JointReference &reference) {
  const uint8_t next_index = 1 - active_index_.load(std::memory_order_relaxed);
  buffers_[next_index] = reference;
  active_index_.store(next_index, std::memory_order_release);
  valid_.store(true, std::memory_order_release);
}

void JointReferenceHandle::clear() { valid_.store(false, std::memory_order_release); }

bool JointReferenceHandle::hasReference() const { return valid_.load(std::memory_order_acquire); }

JointReference JointReferenceHandle::get() const {
  const uint8_t active_index = active_index_.load(std::memory_order_acquire);
  return buffers_[active_index];
}

JointImpedanceTrackingMotion::JointImpedanceTrackingMotion(std::shared_ptr<JointReferenceHandle> reference_handle)
    : JointImpedanceTrackingMotion(std::move(reference_handle), Params{}) {}

JointImpedanceTrackingMotion::JointImpedanceTrackingMotion(
    std::shared_ptr<JointReferenceHandle> reference_handle, const Params &params)
    : JointImpedanceBase(Vector7d::Zero(), Vector7d::Zero(), params),
      reference_handle_(std::move(reference_handle)) {}

JointImpedanceTrackingMotion::JointImpedanceTrackingMotion(ReferenceCallback reference_callback)
    : JointImpedanceTrackingMotion(std::move(reference_callback), Params{}) {}

JointImpedanceTrackingMotion::JointImpedanceTrackingMotion(ReferenceCallback reference_callback, const Params &params)
    : JointImpedanceBase(Vector7d::Zero(), Vector7d::Zero(), params),
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

  if (reference_callback_) {
    reference = reference_callback_(robot_state, time_step, rel_time, abs_time);
  } else if (reference_handle_ && reference_handle_->hasReference()) {
    reference = reference_handle_->get();
  }

  target_ = reference.q;
  target_velocity_ = reference.dq;
  return computeCommand(robot_state, reference);
}

}  // namespace franky
