#include "franky/motion/simple_torque_motion.hpp"

#include <array>
#include <string>

#include "franky/model.hpp"
#include "franky/robot.hpp"

namespace franky {

SimpleTorqueMotion::SimpleTorqueMotion(const SimpleTorqueParams &params)
    : params_(params), torque_handle_(TorqueCommand{params.initial_torque, 0}) {
  params_.validate();
}

SimpleTorqueMotion::SimpleTorqueMotion(const Vector7d &initial_torque, std::optional<double> signal_timeout)
    : SimpleTorqueMotion(SimpleTorqueParams{.initial_torque = initial_torque, .signal_timeout = signal_timeout}) {}

void SimpleTorqueMotion::initImpl(
    const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) {
  current_torque_ = params_.initial_torque;
  last_signal_time_ = 0.0;
}

franka::Torques SimpleTorqueMotion::nextCommandImpl(
    const RobotState &robot_state, franka::Duration /*time_step*/, franka::Duration rel_time,
    franka::Duration /*abs_time*/, const std::optional<franka::Torques> & /*previous_command*/) {
  const double t = rel_time.toSec();

  const auto command = torque_handle_.getUnsafe();
  if (command.seq != last_seq_) {
    last_seq_ = command.seq;
    current_torque_ = command.tau;
    last_signal_time_ = t;
  } else if (params_.signal_timeout.has_value() && t - last_signal_time_ > params_.signal_timeout.value()) {
    throw TorqueSignalTimeoutException(
        "SimpleTorqueMotion did not receive a new torque signal for " + std::to_string(t - last_signal_time_) +
        "s, which exceeds the signal timeout of " + std::to_string(params_.signal_timeout.value()) + "s.");
  }

  Vector7d tau_d = current_torque_;

  tau_d += computeFrictionCompensation(robot_state.dq, params_.friction);
  if (params_.safety.lower_joint_limits.has_value() && params_.safety.upper_joint_limits.has_value()) {
    tau_d += franky::computeJointLimitTorque(
        robot_state.q,
        robot_state.dq,
        params_.safety.lower_joint_limits.value(),
        params_.safety.upper_joint_limits.value(),
        params_.safety.joint_limit_activation_distance,
        params_.safety.joint_limit_stiffness,
        params_.safety.joint_limit_damping,
        params_.safety.joint_limit_max_torque);
  }
  if (params_.compensate_coriolis) tau_d += robot()->model()->coriolis(robot_state);
  tau_d = franky::saturateTorqueRate(tau_d, robot_state.tau_J_d, params_.safety.max_delta_tau);

  std::array<double, 7> tau_array{};
  Eigen::VectorXd::Map(tau_array.data(), 7) = tau_d;
  return franka::Torques(tau_array);
}

}  // namespace franky
