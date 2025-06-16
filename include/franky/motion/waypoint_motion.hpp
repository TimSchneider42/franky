#pragma once

#include <optional>
#include <ruckig/ruckig.hpp>

#include "franky/motion/motion.hpp"

namespace franky {

/**
 * @brief Exception thrown if the motion planner fails.
 */
struct MotionPlannerException : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/**
 * @brief A waypoint with a target and optional parameters.
 *
 * @tparam TargetType The type of the target.
 *
 * @param target                    The target of this waypoint.
 * @param reference_type            The reference type (absolute or relative).
 * @param relative_dynamics_factor  The relative dynamics factor for this
 * waypoint. This factor will get multiplied with the robot's global dynamics
 * factor and the motion dynamics factor to get the actual dynamics factor for
 * this waypoint.
 * @param minimum_time              The minimum time to get to the next
 * waypoint.
 * @param hold_target_duration      For how long to hold the target of this
 * waypoint after it has been reached.
 * @param max_total_duration        The maximum time to try reaching this
 * waypoint before moving on to the next waypoint. Default is infinite.
 */
template <typename TargetType>
struct Waypoint {
  TargetType target;

  RelativeDynamicsFactor relative_dynamics_factor{1.0};

  std::optional<franka::Duration> minimum_time{std::nullopt};

  franka::Duration hold_target_duration{0};

  std::optional<franka::Duration> max_total_duration{std::nullopt};
};

/**
 * @brief A motion following multiple waypoints in a time-optimal way. Works
 * with arbitrary initial conditions.
 * @tparam ControlSignalType    The type of the control signal. Either
 * franka::Torques, franka::JointVelocities, franka::CartesianVelocities,
 * franka::JointPositions or franka::CartesianPose.
 * @tparam WaypointType         The type of the waypoints. Must subclass
 * Waypoint<TargetType>.
 * @tparam TargetType           The type of the target of the waypoints.
 */
template <typename ControlSignalType, typename WaypointType, typename TargetType>
class WaypointMotion : public Motion<ControlSignalType> {
  static_assert(
      std::is_base_of_v<Waypoint<TargetType>, WaypointType>, "WaypointType must inherit from Waypoint<TargetType>");

 public:
  /**
   * @param waypoints                The waypoints to follow.
   * @param return_when_finished     Whether to end the motion when the last
   * waypoint is reached or keep holding the last target.
   */
  explicit WaypointMotion(std::vector<WaypointType> waypoints, bool return_when_finished = true)
      : waypoints_(std::move(waypoints)), return_when_finished_(return_when_finished), prev_result_() {}

 protected:
  void initImpl(const RobotState &robot_state, const std::optional<ControlSignalType> &previous_command) override {
    target_reached_time_ = std::nullopt;
    waypoint_started_time_ = franka::Duration(0);

    waypoint_iterator_ = waypoints_.begin();
    initWaypointMotion(robot_state, previous_command, input_parameter_);
    if (waypoint_iterator_ != waypoints_.end()) {
      checkWaypoint(*waypoint_iterator_);
      setNewWaypoint(robot_state, previous_command, *waypoint_iterator_, input_parameter_);
      setInputLimits(*waypoint_iterator_, input_parameter_);
      prev_result_ = ruckig::Result::Working;
    } else {
      prev_result_ = ruckig::Result::Finished;
    }
  }

  ControlSignalType nextCommandImpl(
      const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
      const std::optional<ControlSignalType> &previous_command) override {
    const auto expected_time_step = franka::Duration(1);
    if (time_step > expected_time_step) {
      // In this case, we missed a couple of steps for some reason. Hence,
      // extrapolate the way the robot does if it does not receive data
      // (constant acceleration model). See
      // https://frankaemika.github.io/docs/libfranka.html#under-the-hood
      extrapolateMotion(robot_state, time_step - expected_time_step, input_parameter_, output_parameter_);
      output_parameter_.pass_to_input(input_parameter_);
    }

    auto max_time_reached = false;
    if (waypoint_iterator_ != waypoints_.end() && waypoint_iterator_->max_total_duration.has_value()) {
      max_time_reached = rel_time - waypoint_started_time_ >= waypoint_iterator_->max_total_duration.value();
    }

    if (prev_result_ == ruckig::Finished || max_time_reached) {
      if (!target_reached_time_.has_value()) {
        target_reached_time_ = rel_time;
      }
      if (waypoint_iterator_ != waypoints_.end()) {
        auto hold_target_duration = waypoint_iterator_->hold_target_duration;
        if (waypoint_iterator_ + 1 == waypoints_.end()) {
          // Allow cooldown of motion, so that the low-pass filter has time to
          // adjust to target values
          hold_target_duration = std::max(hold_target_duration, franka::Duration(10));
        }
        if (rel_time - target_reached_time_.value() >= hold_target_duration || max_time_reached) {
          target_reached_time_ = std::nullopt;
          ++waypoint_iterator_;
          if (waypoint_iterator_ != waypoints_.end()) {
            checkWaypoint(*waypoint_iterator_);
            setNewWaypoint(robot_state, previous_command, *waypoint_iterator_, input_parameter_);
            setInputLimits(*waypoint_iterator_, input_parameter_);
            waypoint_started_time_ = rel_time;
          }
        }
      }
      if (waypoint_iterator_ == waypoints_.end()) {
        auto command = getControlSignal(robot_state, time_step, previous_command, input_parameter_);
        if (!return_when_finished_) return command;
        return franka::MotionFinished(command);
      }
    }

    assert(waypoint_iterator_ != waypoints_.end());

    prev_result_ = trajectory_generator_.update(input_parameter_, output_parameter_);
    if (prev_result_ == ruckig::Result::Working || prev_result_ == ruckig::Result::Finished) {
      // This is a workaround to prevent NaNs from popping up. Must be some bug
      // in ruckig.
      for (int i = 0; i < 7; i++) {
        if (!input_parameter_.enabled[i]) {
          output_parameter_.new_position[i] = 0.0;
          output_parameter_.new_velocity[i] = 0.0;
          output_parameter_.new_acceleration[i] = 0.0;
        }
      }
      output_parameter_.pass_to_input(input_parameter_);
    } else {
      throw MotionPlannerException("Motion planner failed with error code " + std::to_string(prev_result_));
    }

    return getControlSignal(robot_state, time_step, previous_command, input_parameter_);
  };

  virtual void initWaypointMotion(
      const RobotState &robot_state, const std::optional<ControlSignalType> &previous_command,
      ruckig::InputParameter<7> &input_parameter) = 0;

  virtual void setNewWaypoint(
      const RobotState &robot_state, const std::optional<ControlSignalType> &previous_command,
      const WaypointType &new_waypoint, ruckig::InputParameter<7> &input_parameter) = 0;

  virtual void extrapolateMotion(
      const RobotState &robot_state, const franka::Duration &time_step,
      const ruckig::InputParameter<7> &input_parameter, ruckig::OutputParameter<7> &output_parameter) const = 0;

  virtual void checkWaypoint(const WaypointType &waypoint) const {}

  [[nodiscard]] virtual std::tuple<Vector7d, Vector7d, Vector7d> getAbsoluteInputLimits() const = 0;

  [[nodiscard]] virtual ControlSignalType getControlSignal(
      const RobotState &robot_state, const franka::Duration &time_step,
      const std::optional<ControlSignalType> &previous_command, const ruckig::InputParameter<7> &input_parameter) = 0;

  virtual void setInputLimits(const WaypointType &waypoint, ruckig::InputParameter<7> &input_parameter) const = 0;

 private:
  std::vector<WaypointType> waypoints_;
  bool return_when_finished_{true};

  ruckig::Ruckig<7> trajectory_generator_{Robot::control_rate};
  ruckig::Result prev_result_;
  ruckig::InputParameter<7> input_parameter_;
  ruckig::OutputParameter<7> output_parameter_;

  typename std::vector<WaypointType>::iterator waypoint_iterator_;

  std::optional<franka::Duration> target_reached_time_;
  franka::Duration waypoint_started_time_;
};

}  // namespace franky
