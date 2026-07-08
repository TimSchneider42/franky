#pragma once

#include <memory>

#include "franky/motion/motion.hpp"
#include "franky/motion/torque_control_utils.hpp"
#include "franky/motion/wait_free_triple_buffer.hpp"
#include "franky/types.hpp"

namespace franky {

inline Vector7d defaultJointImpedanceStiffness() { return Vector7d::Constant(50.0); }

inline Vector7d defaultJointImpedanceDamping(const Vector7d &stiffness) { return 2.0 * stiffness.cwiseSqrt(); }

inline Vector7d defaultJointImpedanceDamping() {
  return defaultJointImpedanceDamping(defaultJointImpedanceStiffness());
}

/**
 * @brief Joint-space reference for joint impedance motions.
 *
 * The impedance controller tracks the joint position and velocity reference and
 * adds the optional per-cycle feedforward torque term to the commanded torques.
 */
struct JointReference {
  Vector7d q{Vector7d::Zero()};
  Vector7d dq{Vector7d::Zero()};
  Vector7d tau_ff{Vector7d::Zero()};
};

struct JointImpedanceGains {
  JointImpedanceGains() = default;

  explicit JointImpedanceGains(
      const std::optional<Vector7d> &stiffness, const std::optional<Vector7d> &damping = std::nullopt)
      : stiffness(stiffness.value_or(defaultJointImpedanceStiffness())),
        damping(damping.has_value() ? *damping : defaultJointImpedanceDamping(this->stiffness)) {
    validate();
  }

  Vector7d stiffness{defaultJointImpedanceStiffness()};
  Vector7d damping{defaultJointImpedanceDamping()};

  /** @brief Throw std::invalid_argument if any gain is negative or non-finite. */
  void validate() const {
    validateNonNegativeFinite(stiffness, "stiffness");
    validateNonNegativeFinite(damping, "damping");
  }
};

/**
 * @brief Parameters for joint impedance motions.
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

  /** @brief Throw std::invalid_argument if any parameter is out of range. */
  void validate() const {
    validateNonNegativeFinite(stiffness, "stiffness");
    validateNonNegativeFinite(damping, "damping");
    friction.validate();
  }
};

/**
 * @brief Base class for client-side joint impedance motions.
 *
 * This class computes joint torques from a joint-space spring-damper law plus
 * optional torque offset/model compensation. Subclasses implement
 * nextCommandImpl and call computeCommand with their current reference.
 */
class JointImpedanceBase : public Motion<franka::Torques> {
 public:
  [[nodiscard]] const Vector7d &target() const { return target_; }
  [[nodiscard]] const Vector7d &target_velocity() const { return target_velocity_; }
  [[nodiscard]] const JointImpedanceParams &params() const { return params_; }

  void setGains(const JointImpedanceGains &gains) {
    gains.validate();
    gains_handle_.set(gains);
  }
  [[nodiscard]] JointImpedanceGains getGains() const { return gains_handle_.getLastWritten(); }

 protected:
  explicit JointImpedanceBase(
      const Vector7d &target, const Vector7d &target_velocity, const JointImpedanceParams &params,
      double gains_time_constant = 0.1);

  [[nodiscard]] franka::Torques computeCommand(
      const RobotState &robot_state, const JointReference &reference, double dt);

  JointImpedanceParams params_;
  Vector7d target_;
  Vector7d target_velocity_;

 private:
  WaitFreeTripleBuffer<JointImpedanceGains> gains_handle_;
  double gains_time_constant_;
  Vector7d current_stiffness_;
  Vector7d current_damping_;
};

}  // namespace franky
