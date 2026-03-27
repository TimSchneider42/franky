#include "franky/motion/joint_impedance_motion.hpp"
#include "franky/motion/joint_impedance_tracking_motion.hpp"
#include "franky/motion/torque_control_utils.hpp"

#include <array>
#include <cmath>

#include "franky/model.hpp"

namespace franky {

JointImpedanceBase::JointImpedanceBase(
    const Vector7d &target, const Vector7d &target_velocity, const JointImpedanceParams &params,
    std::shared_ptr<JointImpedanceGainsHandle> gains_handle, double gains_time_constant)
    : Motion<franka::Torques>(),
      params_(params),
      target_(target),
      target_velocity_(target_velocity),
      gains_handle_(std::move(gains_handle)),
      gains_time_constant_(gains_time_constant),
      current_stiffness_(params.stiffness),
      current_damping_(params.damping) {}

JointImpedanceMotion::JointImpedanceMotion(const Vector7d &target) : JointImpedanceMotion(target, Params{}) {}

JointImpedanceMotion::JointImpedanceMotion(const Vector7d &target, const Params &params)
    : JointImpedanceMotion(target, Vector7d::Zero(), params) {}

JointImpedanceMotion::JointImpedanceMotion(const Vector7d &target, const Vector7d &target_velocity)
    : JointImpedanceMotion(target, target_velocity, Params{}) {}

JointImpedanceMotion::JointImpedanceMotion(
    const Vector7d &target, const Vector7d &target_velocity, const Params &params)
    : JointImpedanceBase(target, target_velocity, params) {}

franka::Torques JointImpedanceMotion::nextCommandImpl(
    const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
    const std::optional<franka::Torques> &previous_command) {
  JointReference reference;
  reference.q = target_;
  reference.dq = target_velocity_;
  const double dt = time_step.toSec();
  return computeCommand(robot_state, reference, dt);
}

franka::Torques JointImpedanceBase::computeCommand(
    const RobotState &robot_state, const JointReference &reference, double dt) {
  // If a gains handle is present, interpolate toward the target gains.
  if (gains_handle_ && gains_handle_->hasGains()) {
    const auto target_gains = gains_handle_->get();
    const double alpha = 1.0 - std::exp(-dt / gains_time_constant_);
    current_stiffness_ += alpha * (target_gains.stiffness - current_stiffness_);
    current_damping_ += alpha * (target_gains.damping - current_damping_);
  }

  Vector7d torque_feedforward = params_.constant_torque_offset + reference.tau_ff;
  Vector7d tau_d =
      current_stiffness_.asDiagonal() * (reference.q - robot_state.q) +
      current_damping_.asDiagonal() * (reference.dq - robot_state.dq) + torque_feedforward;

  if (params_.joint_limit_repulsion_active) {
    tau_d += franky::computeJointLimitTorque(
        robot_state.q,
        robot_state.dq,
        params_.lower_joint_limits,
        params_.upper_joint_limits,
        params_.joint_limit_activation_distance,
        params_.joint_limit_stiffness,
        params_.joint_limit_damping,
        params_.joint_limit_max_torque);
  }

  auto model = robot()->model();
  if (params_.compensate_coriolis) tau_d += model->coriolis(robot_state);
  tau_d = franky::saturateTorqueRate(tau_d, robot_state.tau_J_d, params_.max_delta_tau);

  std::array<double, 7> tau_d_array{};
  Eigen::VectorXd::Map(tau_d_array.data(), 7) = tau_d;
  return franka::Torques(tau_d_array);
}

}  // namespace franky
