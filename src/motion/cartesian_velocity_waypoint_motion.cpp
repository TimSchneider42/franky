#include "franky/motion/cartesian_velocity_waypoint_motion.hpp"

#include <ruckig/ruckig.hpp>
#include <utility>

#include "franky/cartesian_state.hpp"
#include "franky/robot.hpp"
#include "franky/util.hpp"

namespace franky {

CartesianVelocityWaypointMotion::CartesianVelocityWaypointMotion(
    const std::vector<VelocityWaypoint<RobotVelocity>> &waypoints,
    const RelativeDynamicsFactor &relative_dynamics_factor, Affine ee_frame)
    : VelocityWaypointMotion(waypoints, relative_dynamics_factor), ee_frame_(std::move(ee_frame)) {}

void CartesianVelocityWaypointMotion::checkWaypoint(const VelocityWaypoint<RobotVelocity> &waypoint) const {
  auto [vel_lim, acc_lim, jerk_lim] = getAbsoluteInputLimits();
  if ((waypoint.target.vector_repr().array().abs() > vel_lim.array()).any()) {
    std::stringstream ss;
    ss << "Waypoint velocity " << waypoint.target.vector_repr() << " exceeds maximum velocity " << vel_lim << ".";
    throw std::runtime_error(ss.str());
  }
}

void CartesianVelocityWaypointMotion::initWaypointMotion(
    const RobotState &robot_state, const std::optional<franka::CartesianVelocities> &previous_command,
    ruckig::InputParameter<7> &input_parameter) {
  RobotVelocity current_velocity;
  if (previous_command.has_value()) {
    current_velocity = RobotVelocity(previous_command.value());
  } else {
    current_velocity = RobotVelocity(robot_state.O_dP_EE_c, robot_state.delbow_c);
  }

  Vector7d initial_acceleration_with_elbow =
      (Vector7d() << robot_state.O_ddP_EE_c.vector_repr(), robot_state.ddelbow_c).finished();

  input_parameter.current_position = toStdD<7>(current_velocity.vector_repr());
  input_parameter.current_velocity = toStdD<7>(initial_acceleration_with_elbow);
  input_parameter.current_acceleration = toStdD<7>(Vector7d::Zero());

  if (previous_command.has_value())
    last_elbow_pos_ = previous_command.value().elbow[0];
  else
    last_elbow_pos_ = robot_state.elbow_c.joint_3_pos();
  last_elbow_vel_ = robot_state.delbow_c;
}

franka::CartesianVelocities CartesianVelocityWaypointMotion::getControlSignal(
    const RobotState &robot_state, const franka::Duration &time_step,
    const std::optional<franka::CartesianVelocities> &previous_command,
    const ruckig::InputParameter<7> &input_parameter) {
  auto has_elbow = input_parameter.enabled[6];
  RobotVelocity target_vel(toEigenD<7>(input_parameter.current_position));
  if (has_elbow) {
    auto time_step_s = time_step.toSec();
    auto current_elbow_vel = input_parameter.current_position[6];
    auto current_elbow_acc = input_parameter.current_velocity[6];
    auto last_elbow_acc = 2 * (current_elbow_vel - last_elbow_vel_) / time_step_s - current_elbow_acc;
    auto elbow_jerk = (current_elbow_acc - last_elbow_acc) / time_step_s;
    auto current_elbow_pos = last_elbow_pos_ + current_elbow_vel * time_step_s +
                             0.5 * current_elbow_acc * std::pow(time_step_s, 2) +
                             1.0 / 6.0 * elbow_jerk * std::pow(time_step_s, 3);

    last_elbow_vel_ = input_parameter.current_position[0];
    last_elbow_pos_ = current_elbow_pos;
    auto current_elbow_flip = robot_state.elbow.joint_4_flip();
    if (previous_command.has_value() && previous_command->hasElbow()) {
      current_elbow_flip = static_cast<FlipDirection>(previous_command->elbow[1]);
    }
    ElbowState elbow_state(current_elbow_pos, current_elbow_flip);
    return target_vel.as_franka_velocity(elbow_state);
  }
  return target_vel.as_franka_velocity();
}

void CartesianVelocityWaypointMotion::setNewWaypoint(
    const RobotState &robot_state, const std::optional<franka::CartesianVelocities> &previous_command,
    const VelocityWaypoint<RobotVelocity> &new_waypoint, ruckig::InputParameter<7> &input_parameter) {
  auto new_target_transformed = new_waypoint.target.changeEndEffectorFrame(ee_frame_.inverse().translation());
  // This is a bit of an oversimplification, as the angular velocities don't
  // work like linear velocities (but we pretend they do here). However, it is
  // probably good enough here.
  input_parameter.target_position = toStdD<7>(new_target_transformed.vector_repr());
  input_parameter.target_velocity = toStdD<7>(Vector7d::Zero());
  input_parameter.enabled = {true, true, true, true, true, true, new_target_transformed.elbow_velocity().has_value()};
}

std::tuple<Vector7d, Vector7d, Vector7d> CartesianVelocityWaypointMotion::getAbsoluteInputLimits() const {
  const auto r = robot();
  return {
      vec_cart_rot_elbow(
          r->translation_velocity_limit.get(), r->rotation_velocity_limit.get(), r->elbow_velocity_limit.get()),
      vec_cart_rot_elbow(
          r->translation_acceleration_limit.get(),
          r->rotation_acceleration_limit.get(),
          r->elbow_acceleration_limit.get()),
      vec_cart_rot_elbow(r->translation_jerk_limit.get(), r->rotation_jerk_limit.get(), r->elbow_jerk_limit.get())};
}

std::tuple<Vector7d, Vector7d, Vector7d> CartesianVelocityWaypointMotion::getDesiredState(
    const RobotState &robot_state) const {
  RobotVelocity current_velocity(robot_state.O_dP_EE_d, robot_state.delbow_c);
  Vector7d current_acceleration =
      (Vector7d() << robot_state.O_ddP_EE_c.vector_repr(), robot_state.ddelbow_c).finished();
  return {current_velocity.vector_repr(), current_acceleration, Vector7d::Zero()};
}

}  // namespace franky
