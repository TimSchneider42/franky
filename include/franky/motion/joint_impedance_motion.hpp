#pragma once

#include <optional>

#include "franky/motion/motion.hpp"
#include "franky/types.hpp"

namespace franky {

/**
 * @brief Client-side joint impedance controller.
 *
 * This motion uses Franka's torque interface and computes joint torques from a
 * joint-space spring-damper law plus optional feedforward/model compensation.
 */
class JointImpedanceMotion : public Motion<franka::Torques> {
 public:
  struct Params {
    /** Joint stiffness gains in [Nm/rad]. */
    Vector7d stiffness{Vector7d::Constant(50.0)};

    /** Joint damping gains in [Nms/rad]. */
    Vector7d damping{Vector7d::Constant(10.0)};

    /** Additional feedforward torques in [Nm]. */
    Vector7d torque_feedforward{Vector7d::Zero()};

    /** Compensate Coriolis forces using the robot model. */
    bool compensate_coriolis{true};

    /** Maximum allowed torque step per cycle in [Nm]. */
    double max_delta_tau{1.0};

    /** Whether to enable joint-limit repulsion torques. */
    bool joint_limit_repulsion_active{false};

    /** Lower soft joint limits in [rad]. */
    Vector7d lower_joint_limits{Vector7d::Zero()};

    /** Upper soft joint limits in [rad]. */
    Vector7d upper_joint_limits{Vector7d::Zero()};

    /** Activation distance from a limit in [rad]. */
    double joint_limit_activation_distance{0.1};

    /** Base repulsion gain in [Nm]. */
    double joint_limit_stiffness{4.0};

    /** Additional damping when moving into a limit in [Nms/rad]. */
    double joint_limit_damping{1.0};

    /** Absolute torque clamp for the repulsion term in [Nm]. */
    double joint_limit_max_torque{5.0};
  };

  explicit JointImpedanceMotion(const Vector7d &target);
  explicit JointImpedanceMotion(const Vector7d &target, const Params &params);
  JointImpedanceMotion(const Vector7d &target, const Vector7d &target_velocity);
  JointImpedanceMotion(const Vector7d &target, const Vector7d &target_velocity, const Params &params);

  [[nodiscard]] const Vector7d &target() const { return target_; }
  [[nodiscard]] const Vector7d &target_velocity() const { return target_velocity_; }

 protected:
  void initImpl(const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) override;

  franka::Torques nextCommandImpl(
      const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
      const std::optional<franka::Torques> &previous_command) override;

 private:
  [[nodiscard]] Vector7d computeJointLimitTorque(const RobotState &robot_state) const;
  [[nodiscard]] Vector7d saturateTorqueRate(const Vector7d &tau_d_calculated, const RobotState &robot_state) const;

  Vector7d target_;
  Vector7d target_velocity_;
  Params params_;
};

}  // namespace franky
