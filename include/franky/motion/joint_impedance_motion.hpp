#pragma once

#include <memory>
#include <optional>

#include "franky/motion/impedance_gains_handle.hpp"
#include "franky/motion/motion.hpp"
#include "franky/types.hpp"

namespace franky {

struct JointReference;

struct TorqueSafetyParams {
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

/**
 * @brief Client-side joint impedance controller.
 *
 * This motion uses Franka's torque interface and computes joint torques from a
 * joint-space spring-damper law plus optional torque offset/model compensation.
 */
struct JointImpedanceParams : public TorqueSafetyParams {
  /** Joint stiffness gains in [Nm/rad]. */
  Vector7d stiffness{defaultJointImpedanceStiffness()};

  /** Joint damping gains in [Nms/rad]. */
  Vector7d damping{defaultJointImpedanceDamping()};

  /** Constant torque offset added to every command in [Nm]. */
  Vector7d constant_torque_offset{Vector7d::Zero()};

  /** Compensate Coriolis forces using the robot model. */
  bool compensate_coriolis{true};
};

class JointImpedanceBase : public Motion<franka::Torques> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  [[nodiscard]] const Vector7d &target() const { return target_; }
  [[nodiscard]] const Vector7d &target_velocity() const { return target_velocity_; }

 protected:
  explicit JointImpedanceBase(
      const Vector7d &target, const Vector7d &target_velocity, const JointImpedanceParams &params,
      std::shared_ptr<JointImpedanceGainsHandle> gains_handle = nullptr, double gains_time_constant = 0.1);

  [[nodiscard]] franka::Torques computeCommand(
      const RobotState &robot_state, const JointReference &reference, double dt);

  JointImpedanceParams params_;
  Vector7d target_;
  Vector7d target_velocity_;

 private:
  std::shared_ptr<JointImpedanceGainsHandle> gains_handle_;
  double gains_time_constant_;
  Vector7d current_stiffness_;
  Vector7d current_damping_;
};

class JointImpedanceMotion : public JointImpedanceBase {
 public:
  using Params = JointImpedanceParams;

  explicit JointImpedanceMotion(const Vector7d &target);
  explicit JointImpedanceMotion(const Vector7d &target, const Params &params);
  JointImpedanceMotion(const Vector7d &target, const Vector7d &target_velocity);
  JointImpedanceMotion(const Vector7d &target, const Vector7d &target_velocity, const Params &params);

 protected:
  franka::Torques nextCommandImpl(
      const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
      const std::optional<franka::Torques> &previous_command) override;
};

}  // namespace franky
