#pragma once

#include <franka/robot_state.h>

#include <atomic>
#include <optional>
#include <ruckig/ruckig.hpp>

#include "franky/joint_state.hpp"
#include "franky/motion/velocity_waypoint_motion.hpp"

namespace franky {

/**
 * @brief Joint velocity waypoint motion.
 *
 * This motion follows multiple joint waypoints in a time-optimal way.
 */
class JointVelocityWaypointMotion : public VelocityWaypointMotion<franka::JointVelocities, Vector7d> {
 public:
  /**
   * @param waypoints Joint waypoints to follow.
   * @param relative_dynamics_factor The relative dynamics factor for this
   * motion. The factor will get multiplied with the robot's global dynamics
   * factor to get the actual dynamics factor for this motion.
   */
  explicit JointVelocityWaypointMotion(
      const std::vector<VelocityWaypoint<Vector7d>> &waypoints,
      const RelativeDynamicsFactor &relative_dynamics_factor = 1.0);

 protected:
  void checkWaypoint(const VelocityWaypoint<Vector7d> &waypoint) const override;

  void initWaypointMotion(
      const RobotState &robot_state, const std::optional<franka::JointVelocities> &previous_command,
      ruckig::InputParameter<7> &input_parameter) override;

  void setNewWaypoint(
      const RobotState &robot_state, const std::optional<franka::JointVelocities> &previous_command,
      const VelocityWaypoint<Vector7d> &new_waypoint, ruckig::InputParameter<7> &input_parameter) override;

  [[nodiscard]] std::tuple<Vector7d, Vector7d, Vector7d> getAbsoluteInputLimits() const override;

  [[nodiscard]] franka::JointVelocities getControlSignal(
      const RobotState &robot_state, const franka::Duration &time_step,
      const std::optional<franka::JointVelocities> &previous_command,
      const ruckig::InputParameter<7> &input_parameter) override;

  [[nodiscard]] std::tuple<Vector7d, Vector7d, Vector7d> getDesiredState(const RobotState &robot_state) const override;
};

}  // namespace franky
