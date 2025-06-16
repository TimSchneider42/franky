#pragma once

#include <franka/robot_state.h>

#include <atomic>
#include <optional>
#include <ruckig/ruckig.hpp>

#include "franky/cartesian_state.hpp"
#include "franky/motion/velocity_waypoint_motion.hpp"
#include "franky/robot.hpp"
#include "franky/robot_pose.hpp"
#include "franky/util.hpp"

namespace franky {

/**
 * @brief Cartesian velocity waypoint motion.
 *
 * This motion follows multiple consecutive cartesian velocity targets in a
 * time-optimal way.
 */
class CartesianVelocityWaypointMotion : public VelocityWaypointMotion<franka::CartesianVelocities, RobotVelocity> {
 public:
  /**
   * @param waypoints                Waypoints to follow.
   * @param relative_dynamics_factor The relative dynamics factor for this
   * motion. This factor will get multiplied with the robot's global dynamics
   * factor to get the actual dynamics factor for this motion.
   * @param ee_frame                 The end-effector frame for which the target
   * is defined. This is a transformation from the configured end-effector frame
   * to the end-effector frame the target is defined for.
   */
  explicit CartesianVelocityWaypointMotion(
      const std::vector<VelocityWaypoint<RobotVelocity>> &waypoints,
      const RelativeDynamicsFactor &relative_dynamics_factor = 1.0, Affine ee_frame = Affine::Identity());

 protected:
  void checkWaypoint(const VelocityWaypoint<RobotVelocity> &waypoint) const override;

  void initWaypointMotion(
      const RobotState &robot_state, const std::optional<franka::CartesianVelocities> &previous_command,
      ruckig::InputParameter<7> &input_parameter) override;

  void setNewWaypoint(
      const RobotState &robot_state, const std::optional<franka::CartesianVelocities> &previous_command,
      const VelocityWaypoint<RobotVelocity> &new_waypoint, ruckig::InputParameter<7> &input_parameter) override;

  [[nodiscard]] std::tuple<Vector7d, Vector7d, Vector7d> getAbsoluteInputLimits() const override;

  [[nodiscard]] franka::CartesianVelocities getControlSignal(
      const RobotState &robot_state, const franka::Duration &time_step,
      const std::optional<franka::CartesianVelocities> &previous_command,
      const ruckig::InputParameter<7> &input_parameter) override;

  [[nodiscard]] std::tuple<Vector7d, Vector7d, Vector7d> getDesiredState(const RobotState &robot_state) const override;

 private:
  Affine ee_frame_;
  double last_elbow_pos_{};
  double last_elbow_vel_{};

  static Vector7d vec_cart_rot_elbow(double cart, double rot, double elbow) {
    return {cart, cart, cart, rot, rot, rot, elbow};
  }
};

}  // namespace franky
