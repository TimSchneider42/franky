#include "franky/motion/joint_impedance_motion.hpp"
#include "franky/motion/joint_impedance_tracking_motion.hpp"
#include "franky/motion/torque_control_utils.hpp"

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
