#include "franky/motion/cartesian_impedance_tracking_motion.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace franky {

void CartesianReferenceHandle::set(const CartesianReference &reference) {
  const uint8_t next_index = 1 - active_index_.load(std::memory_order_relaxed);
  buffers_[next_index] = reference;
  active_index_.store(next_index, std::memory_order_release);
  valid_.store(true, std::memory_order_release);
}

void CartesianReferenceHandle::clear() { valid_.store(false, std::memory_order_release); }

bool CartesianReferenceHandle::hasReference() const { return valid_.load(std::memory_order_acquire); }

CartesianReference CartesianReferenceHandle::get() const {
  const uint8_t active_index = active_index_.load(std::memory_order_acquire);
  return buffers_[active_index];
}

CartesianImpedanceTrackingMotion::CartesianImpedanceTrackingMotion(std::shared_ptr<CartesianReferenceHandle> reference_handle)
    : CartesianImpedanceTrackingMotion(std::move(reference_handle), Params()) {}

CartesianImpedanceTrackingMotion::CartesianImpedanceTrackingMotion(
    std::shared_ptr<CartesianReferenceHandle> reference_handle, const Params &params)
    : CartesianImpedanceBase(Affine::Identity(), params),
      reference_handle_(std::move(reference_handle)),
      target_(Affine::Identity()) {}

CartesianImpedanceTrackingMotion::CartesianImpedanceTrackingMotion(
    std::shared_ptr<CartesianReferenceHandle> reference_handle, const Params &params,
    std::shared_ptr<CartesianImpedanceGainsHandle> gains_handle, double gains_time_constant)
    : CartesianImpedanceBase(Affine::Identity(), params, std::move(gains_handle), gains_time_constant),
      reference_handle_(std::move(reference_handle)),
      target_(Affine::Identity()) {}

CartesianImpedanceTrackingMotion::CartesianImpedanceTrackingMotion(ReferenceCallback reference_callback)
    : CartesianImpedanceTrackingMotion(std::move(reference_callback), Params()) {}

CartesianImpedanceTrackingMotion::CartesianImpedanceTrackingMotion(
    ReferenceCallback reference_callback, const Params &params)
    : CartesianImpedanceBase(Affine::Identity(), params),
      reference_callback_(std::move(reference_callback)),
      target_(Affine::Identity()) {}

void CartesianImpedanceTrackingMotion::initImpl(
    const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) {
  CartesianImpedanceBase::initImpl(robot_state, previous_command);
  target_ = Affine(Eigen::Matrix4d::Map(robot_state.O_T_EE_c.data()));
  target_twist_ = std::nullopt;
  if (reference_handle_ && reference_handle_->hasReference()) {
    auto reference = reference_handle_->get();
    target_ = reference.target;
    target_twist_ = reference.target_twist;
  } else if (reference_callback_) {
    auto reference = reference_callback_(robot_state, franka::Duration(0), franka::Duration(0), franka::Duration(0));
    target_ = reference.target;
    target_twist_ = reference.target_twist;
  }
}

std::tuple<CartesianReference, bool> CartesianImpedanceTrackingMotion::update(
    const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time) {
  if (reference_callback_) {
    auto reference = reference_callback_(robot_state, time_step, rel_time, abs_time);
    target_ = reference.target;
    target_twist_ = reference.target_twist;
  } else if (reference_handle_ && reference_handle_->hasReference()) {
    auto reference = reference_handle_->get();
    target_ = reference.target;
    target_twist_ = reference.target_twist;
  }
  return {CartesianReference{target_, target_twist_}, false};
}

}  // namespace franky
