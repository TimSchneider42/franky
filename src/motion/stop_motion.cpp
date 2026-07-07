#include "franky/motion/stop_motion.hpp"

#include <algorithm>
#include <array>

#include "franky/model.hpp"
#include "franky/robot.hpp"

namespace franky {

void StopMotion<franka::Torques>::initImpl(
    const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) {
  tau_start_ = previous_command.has_value() ? Vector7d(Eigen::Map<const Vector7d>(previous_command->tau_J.data()))
                                            : robot_state.tau_J_d;
}

franka::Torques StopMotion<franka::Torques>::nextCommandImpl(
    const RobotState &robot_state, franka::Duration /*time_step*/, franka::Duration rel_time,
    franka::Duration /*abs_time*/, const std::optional<franka::Torques> & /*previous_command*/) {
  const double t = rel_time.toSec();

  Vector7d tau_damp = -(params_.damping.asDiagonal() * robot_state.dq);
  if (params_.compensate_coriolis) tau_damp += robot()->model()->coriolis(robot_state);

  const double s = params_.ramp_duration > 0.0 ? std::clamp(t / params_.ramp_duration, 0.0, 1.0) : 1.0;
  Vector7d tau_d = (1.0 - s) * tau_start_ + s * tau_damp;

  tau_d = franky::saturateTorqueRate(tau_d, robot_state.tau_J_d, params_.max_delta_tau);

  std::array<double, 7> tau_array{};
  Eigen::VectorXd::Map(tau_array.data(), 7) = tau_d;

  const bool ramp_done = s >= 1.0;
  const bool at_rest = robot_state.dq.cwiseAbs().maxCoeff() < params_.velocity_epsilon;
  if ((ramp_done && at_rest) || t >= params_.max_duration) {
    return franka::MotionFinished(franka::Torques(tau_array));
  }
  return franka::Torques(tau_array);
}

}  // namespace franky
