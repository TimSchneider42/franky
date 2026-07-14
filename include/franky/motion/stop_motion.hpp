#pragma once

#include <franka/control_types.h>

#include <optional>

#include "franky/motion/cartesian_motion.hpp"
#include "franky/motion/cartesian_velocity_motion.hpp"
#include "franky/motion/joint_impedance_base.hpp"
#include "franky/motion/joint_motion.hpp"
#include "franky/motion/joint_velocity_motion.hpp"
#include "franky/motion/motion.hpp"
#include "franky/motion/reference_type.hpp"
#include "franky/motion/torque_control_utils.hpp"
#include "franky/robot_state.hpp"
#include "franky/types.hpp"

namespace franky {

template <typename ControlSignalType>
class StopMotion;

/**
 * @brief Stop motion for joint position control mode.
 */
template <>
class StopMotion<franka::JointPositions> : public JointMotion {
 public:
  /**
   *
   * @param relative_dynamics_factor Relative dynamics factor for this stop
   * motion.
   */
  explicit StopMotion(const RelativeDynamicsFactor &relative_dynamics_factor = 1.0)
      : JointMotion(JointState(Vector7d::Zero()), ReferenceType::kRelative, relative_dynamics_factor) {}
};

/**
 * @brief Stop motion for joint velocity control mode.
 */
template <>
class StopMotion<franka::JointVelocities> : public JointVelocityMotion {
 public:
  /**
   *
   * @param relative_dynamics_factor Relative dynamics factor for this stop
   * motion.
   */
  explicit StopMotion(const RelativeDynamicsFactor &relative_dynamics_factor = 1.0)
      : JointVelocityMotion(Vector7d::Zero(), franka::Duration(0), relative_dynamics_factor) {}
};

/**
 * @brief Stop motion for cartesian pose control mode.
 */
template <>
class StopMotion<franka::CartesianPose> : public CartesianWaypointMotion {
 public:
  /**
   *
   * @param relative_dynamics_factor Relative dynamics factor for this stop
   * motion.
   */
  explicit StopMotion(const RelativeDynamicsFactor &relative_dynamics_factor = 1.0)
      : CartesianWaypointMotion(
            {PositionWaypoint<CartesianState>{
                {
                    .target = RobotPose(),
                    .hold_target_duration = franka::Duration(50),
                },
                ReferenceType::kRelative}},
            relative_dynamics_factor) {}
};

/**
 * @brief Stop motion for cartesian velocity control mode.
 */
template <>
class StopMotion<franka::CartesianVelocities> : public CartesianVelocityMotion {
 public:
  /**
   *
   * @param relative_dynamics_factor Relative dynamics factor for this stop
   * motion.
   */
  explicit StopMotion(const RelativeDynamicsFactor &relative_dynamics_factor = 1.0)
      : CartesianVelocityMotion(RobotVelocity(), franka::Duration(0), relative_dynamics_factor) {}
};

/**
 * @brief Parameters for the torque-control stop motion (StopMotion<franka::Torques>).
 */
struct TorqueStopParams {
  /** Pure joint damping [Nms/rad] used to bring the arm to rest. */
  Vector7d damping{defaultJointImpedanceDamping()};

  /**
   * Duration [s] over which the inherited command is ramped into the damping-only law.
   * Set to 0 to switch immediately.
   */
  double ramp_duration{0.2};

  /** Finish once every joint speed is below this [rad/s] (after the ramp completes). */
  double velocity_epsilon{0.02};

  /** Maximum duration [s] before the motion finishes regardless of velocity. */
  double max_duration{2.0};

  /** Compensate Coriolis forces. */
  bool compensate_coriolis{true};

  /** Per-tick torque-rate limit [Nm] for smoothness. */
  double max_delta_tau{1.0};
};

/**
 * @brief Graceful stop for torque-control (impedance) motions.
 *
 * Torque motions never signal MotionFinished on their own, and franka::Robot::stop()
 * preempts the control loop with a franka::ControlException rather than ramping down.
 * This motion ramps the last commanded torque into a damping-only law and then returns
 * franka::MotionFinished.
 */
template <>
class StopMotion<franka::Torques> : public Motion<franka::Torques> {
 public:
  explicit StopMotion(const TorqueStopParams &params = {}) : params_(params) {}

 protected:
  void initImpl(const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) override;

  franka::Torques nextCommandImpl(
      const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
      const std::optional<franka::Torques> &previous_command) override;

 private:
  TorqueStopParams params_;
  // Initial torque command captured in initImpl.
  Vector7d tau_start_{Vector7d::Zero()};
};

}  // namespace franky
