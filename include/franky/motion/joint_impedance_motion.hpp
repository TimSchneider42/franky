#pragma once

#include <memory>
#include <optional>

#include "franky/motion/impedance_gains_handle.hpp"
#include "franky/motion/motion.hpp"
#include "franky/motion/torque_control_utils.hpp"
#include "franky/types.hpp"

namespace franky {

struct JointReference;

/**
 * @brief Client-side joint impedance controller.
 *
 * This motion uses Franka's torque interface and computes joint torques from a
 * joint-space spring-damper law plus optional torque offset/model compensation.
 */
struct JointImpedanceParams {
  /** Joint stiffness gains in [Nm/rad]. */
  Vector7d stiffness{defaultJointImpedanceStiffness()};

  /** Joint damping gains in [Nms/rad]. */
  Vector7d damping{defaultJointImpedanceDamping()};

  /** Constant torque offset added to every command in [Nm]. */
  Vector7d constant_torque_offset{Vector7d::Zero()};

  /** Compensate Coriolis forces using the robot model. */
  bool compensate_coriolis{true};

  /** Shared torque safety limits and soft joint-limit repulsion settings. */
  TorqueSafetyParams safety{};

  /** Joint friction compensation settings. */
  FrictionCompensationParams friction{};
};

class JointImpedanceBase : public Motion<franka::Torques> {
 public:
  [[nodiscard]] const Vector7d &target() const { return target_; }
  [[nodiscard]] const Vector7d &target_velocity() const { return target_velocity_; }
  [[nodiscard]] const JointImpedanceParams &params() const { return params_; }

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
