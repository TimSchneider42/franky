#pragma once

#include <ruckig/ruckig.hpp>

#include "franky/motion/reference_type.hpp"
#include "franky/motion/waypoint_motion.hpp"
#include "franky/relative_dynamics_factor.hpp"
#include "franky/util.hpp"

namespace franky {

/**
 * @brief A velocity waypoint with a target.
 *
 * @tparam TargetType The type of the target.
 */
template <typename TargetType>
using VelocityWaypoint = Waypoint<TargetType>;

/**
 * @brief A motion following multiple positional waypoints in a time-optimal
 * way. Works with arbitrary initial conditions.
 * @tparam ControlSignalType The type of the control signal. Either
 * franka::Torques, franka::JointVelocities, franka::CartesianVelocities,
 * franka::JointPositions or franka::CartesianPose.
 * @tparam TargetType The type of the target of the waypoints.
 */
template <typename ControlSignalType, typename TargetType>
class VelocityWaypointMotion : public WaypointMotion<ControlSignalType, VelocityWaypoint<TargetType>, TargetType> {
 public:
  /**
   * @param waypoints                The waypoints to follow.
   * @param relative_dynamics_factor The relative dynamics factor for this
   * motion. This factor will get multiplied with the robot's global dynamics
   * factor to get the actual dynamics factor for this motion.
   */
  explicit VelocityWaypointMotion(
      std::vector<VelocityWaypoint<TargetType>> waypoints, const RelativeDynamicsFactor &relative_dynamics_factor = 1.0)
      : WaypointMotion<ControlSignalType, VelocityWaypoint<TargetType>, TargetType>(waypoints, true),
        relative_dynamics_factor_(relative_dynamics_factor) {}

 protected:
  void setInputLimits(
      const VelocityWaypoint<TargetType> &waypoint, ruckig::InputParameter<7> &input_parameter) const override {
    auto [vel_lim, acc_lim, jerk_lim] = getAbsoluteInputLimits();

    auto relative_dynamics_factor =
        waypoint.relative_dynamics_factor * relative_dynamics_factor_ * this->robot()->relative_dynamics_factor();

    input_parameter.max_velocity = toStdD<7>(relative_dynamics_factor.acceleration() * acc_lim);
    input_parameter.max_acceleration = toStdD<7>(relative_dynamics_factor.jerk() * jerk_lim);
    input_parameter.max_jerk = toStdD<7>(Vector7d::Constant(std::numeric_limits<double>::infinity()));

    if (relative_dynamics_factor.max_dynamics()) {
      input_parameter.synchronization = ruckig::Synchronization::TimeIfNecessary;
    } else {
      input_parameter.synchronization = ruckig::Synchronization::Time;
      if (waypoint.minimum_time.has_value()) input_parameter.minimum_duration = waypoint.minimum_time.value().toSec();
    }
  }

  void extrapolateMotion(
      const RobotState &robot_state, const franka::Duration &time_step,
      const ruckig::InputParameter<7> &input_parameter, ruckig::OutputParameter<7> &output_parameter) const override {
    auto [vel_lim, acc_lim, jerk_lim] = getAbsoluteInputLimits();

    // We use the desired state here as this is likely what the robot uses
    // internally as well
    auto [vel_d, acc_d, _] = getDesiredState(robot_state);

    auto vel = toEigenD<7>(input_parameter.current_velocity);

    // Retain difference between desired state and motion planner state
    auto vel_diff = vel - vel_d;

    auto new_vel_d = (vel_d + acc_d * time_step.toSec()).cwiseMin(vel_lim).cwiseMax(-vel_lim);

    // Franka assumes a constant acceleration model if no new input is received.
    // See https://frankaemika.github.io/docs/libfranka.html#under-the-hood
    output_parameter.new_acceleration = toStdD<7>(Vector7d::Zero());
    output_parameter.new_velocity = input_parameter.current_velocity;
    output_parameter.new_position = toStdD<7>(new_vel_d + vel_diff);
  }

  [[nodiscard]] std::tuple<Vector7d, Vector7d, Vector7d> getAbsoluteInputLimits() const override = 0;

  [[nodiscard]] virtual std::tuple<Vector7d, Vector7d, Vector7d> getDesiredState(
      const RobotState &robot_state) const = 0;

 private:
  RelativeDynamicsFactor relative_dynamics_factor_;
};

}  // namespace franky
