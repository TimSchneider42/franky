#include "franky/motion/cartesian_impedance_tracking_motion.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace franky {

CartesianImpedanceTrackingMotion::CartesianImpedanceTrackingMotion(const Params &params, double gains_time_constant)
    : CartesianImpedanceBase(Affine::Identity(), params, gains_time_constant) {}

CartesianImpedanceTrackingMotion::CartesianImpedanceTrackingMotion(
    ReferenceCallback reference_callback, const Params &params, double gains_time_constant)
    : CartesianImpedanceBase(Affine::Identity(), params, gains_time_constant),
      reference_callback_(std::move(reference_callback)) {}

void CartesianImpedanceTrackingMotion::initImpl(
    const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) {
  // Seed the fallback target from the measured pose only. nextCommandImpl runs later in this same
  // control cycle and re-reads the handle or callback, overwriting anything prefetched here, so a
  // prefetch would be discarded while still advancing a stateful callback an extra time on every
  // initialization -- including the re-init a reaction triggers when it swaps motions. The handle
  // and callback are therefore polled exactly once per cycle, from nextCommandImpl. Mirrors
  // JointImpedanceTrackingMotion::initImpl.
  target_ = Affine(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  target_twist_ = std::nullopt;
  target_acceleration_ = std::nullopt;
}

franka::Torques CartesianImpedanceTrackingMotion::nextCommandImpl(
    const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
    const std::optional<franka::Torques> &previous_command) {
  CartesianReference reference;
  reference.target = target_;
  reference.target_twist = target_twist_;
  reference.target_acceleration = target_acceleration_;

  auto opt_reference = reference_handle_.getUnsafe();
  if (opt_reference) {
    reference = *opt_reference;
  } else if (reference_callback_) {
    reference = reference_callback_(robot_state, time_step, rel_time, abs_time);
  }

  target_ = reference.target;
  target_twist_ = reference.target_twist;
  target_acceleration_ = reference.target_acceleration;
  const double dt = time_step.toSec();
  return computeCommand(robot_state, reference, dt);
}

}  // namespace franky
