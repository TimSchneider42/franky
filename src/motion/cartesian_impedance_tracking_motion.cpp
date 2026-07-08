#include "franky/motion/cartesian_impedance_tracking_motion.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace franky {

CartesianImpedanceTrackingMotion::CartesianImpedanceTrackingMotion(
    std::shared_ptr<CartesianReferenceHandle> reference_handle)
    : CartesianImpedanceTrackingMotion(std::move(reference_handle), Params()) {}

CartesianImpedanceTrackingMotion::CartesianImpedanceTrackingMotion(
    std::shared_ptr<CartesianReferenceHandle> reference_handle, const Params &params)
    : CartesianImpedanceBase(Affine::Identity(), params), reference_handle_(std::move(reference_handle)) {}

CartesianImpedanceTrackingMotion::CartesianImpedanceTrackingMotion(
    std::shared_ptr<CartesianReferenceHandle> reference_handle, const Params &params,
    std::shared_ptr<CartesianImpedanceGainsHandle> gains_handle, double gains_time_constant)
    : CartesianImpedanceBase(Affine::Identity(), params, std::move(gains_handle), gains_time_constant),
      reference_handle_(std::move(reference_handle)) {}

CartesianImpedanceTrackingMotion::CartesianImpedanceTrackingMotion(ReferenceCallback reference_callback)
    : CartesianImpedanceTrackingMotion(std::move(reference_callback), Params()) {}

CartesianImpedanceTrackingMotion::CartesianImpedanceTrackingMotion(
    ReferenceCallback reference_callback, const Params &params)
    : CartesianImpedanceBase(Affine::Identity(), params), reference_callback_(std::move(reference_callback)) {}

void CartesianImpedanceTrackingMotion::initImpl(
    const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) {
  target_ = Affine(Eigen::Matrix4d::Map(robot_state.O_T_EE_c.data()));
  target_twist_ = std::nullopt;
  target_acceleration_ = std::nullopt;
  if (reference_handle_ && reference_handle_->hasValue()) {
    auto reference = reference_handle_->get();
    target_ = reference.target;
    target_twist_ = reference.target_twist;
    target_acceleration_ = reference.target_acceleration;
  } else if (reference_callback_) {
    auto reference = reference_callback_(robot_state, franka::Duration(0), franka::Duration(0), franka::Duration(0));
    target_ = reference.target;
    target_twist_ = reference.target_twist;
    target_acceleration_ = reference.target_acceleration;
  }
}

franka::Torques CartesianImpedanceTrackingMotion::nextCommandImpl(
    const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
    const std::optional<franka::Torques> &previous_command) {
  CartesianReference reference{target_, target_twist_, target_acceleration_};

  if (reference_callback_) {
    reference = reference_callback_(robot_state, time_step, rel_time, abs_time);
  } else if (reference_handle_ && reference_handle_->hasValue()) {
    reference = reference_handle_->get();
  }

  target_ = reference.target;
  target_twist_ = reference.target_twist;
  target_acceleration_ = reference.target_acceleration;
  const double dt = time_step.toSec();
  return computeCommand(robot_state, reference, dt);
}

}  // namespace franky
