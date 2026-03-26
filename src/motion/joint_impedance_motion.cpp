#include "franky/motion/joint_impedance_motion.hpp"
#include "franky/motion/joint_impedance_tracking_motion.hpp"

#include <array>

#include "franky/model.hpp"

namespace franky {

JointImpedanceBase::JointImpedanceBase(
    const Vector7d &target, const Vector7d &target_velocity, const JointImpedanceParams &params)
    : Motion<franka::Torques>(), params_(params), target_(target), target_velocity_(target_velocity) {}

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
  return computeCommand(robot_state, reference);
}

franka::Torques JointImpedanceBase::computeCommand(const RobotState &robot_state, const JointReference &reference) const {
  Vector7d torque_feedforward = params_.constant_torque_offset + reference.tau_ff;
  Vector7d tau_d =
      params_.stiffness.asDiagonal() * (reference.q - robot_state.q) +
      params_.damping.asDiagonal() * (reference.dq - robot_state.dq) + torque_feedforward;

  if (params_.joint_limit_repulsion_active) tau_d += computeJointLimitTorque(robot_state);

  auto model = robot()->model();
  if (params_.compensate_coriolis) tau_d += model->coriolis(robot_state);
  tau_d = saturateTorqueRate(tau_d, robot_state);

  std::array<double, 7> tau_d_array{};
  Eigen::VectorXd::Map(tau_d_array.data(), 7) = tau_d;
  return franka::Torques(tau_d_array);
}

Vector7d JointImpedanceBase::computeJointLimitTorque(const RobotState &robot_state) const {
  Vector7d tau_limit = Vector7d::Zero();
  const double activation_distance = std::max(params_.joint_limit_activation_distance, 1e-6);

  for (size_t i = 0; i < 7; i++) {
    const double q = robot_state.q[i];
    const double dq = robot_state.dq[i];
    const double lower_soft_limit = params_.lower_joint_limits[i] + activation_distance;
    const double upper_soft_limit = params_.upper_joint_limits[i] - activation_distance;

    if (q < lower_soft_limit) {
      const double penetration = (lower_soft_limit - q) / activation_distance;
      double tau = params_.joint_limit_stiffness * (std::exp(penetration) - 1.0);
      if (dq < 0.0) tau += params_.joint_limit_damping * (-dq);
      tau_limit[i] = std::min(tau, params_.joint_limit_max_torque);
    } else if (q > upper_soft_limit) {
      const double penetration = (q - upper_soft_limit) / activation_distance;
      double tau = -params_.joint_limit_stiffness * (std::exp(penetration) - 1.0);
      if (dq > 0.0) tau -= params_.joint_limit_damping * dq;
      tau_limit[i] = std::max(tau, -params_.joint_limit_max_torque);
    }
  }

  return tau_limit;
}

Vector7d JointImpedanceBase::saturateTorqueRate(const Vector7d &tau_d_calculated, const RobotState &robot_state) const {
  const Vector7d &tau_reference = robot_state.tau_J_d;
  Vector7d tau_d_saturated;
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_reference[i];
    tau_d_saturated[i] =
        tau_reference[i] + std::max(std::min(difference, params_.max_delta_tau), -params_.max_delta_tau);
  }
  return tau_d_saturated;
}

}  // namespace franky
