#pragma once

#include <memory>
#include <optional>

#include "franky/motion/cartesian_impedance_base.hpp"
#include "franky/motion/motion.hpp"
#include "franky/motion/torque_control_utils.hpp"
#include "franky/types.hpp"
#include "franky/wait_free_triple_buffer.hpp"

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
  /** Desired joint positions [rad]. */
  Vector7d q{Vector7d::Zero()};

  /** Desired joint velocities [rad/s]. */
  Vector7d dq{Vector7d::Zero()};

  /** Feedforward torque added to the commanded torques [Nm]. */
  Vector7d tau_ff{Vector7d::Zero()};

  /** @brief Throw std::invalid_argument if any value is non-finite. */
  void validate() const {
    validateFinite(q, "q");
    validateFinite(dq, "dq");
    validateFinite(tau_ff, "tau_ff");
  }
};

/**
 * @brief Runtime-adjustable stiffness and damping gains for joint impedance
 * motions.
 */
struct JointImpedanceGains {
  JointImpedanceGains() = default;

  explicit JointImpedanceGains(
      const std::optional<Vector7d> &stiffness, const std::optional<Vector7d> &damping = std::nullopt)
      : stiffness(stiffness.value_or(defaultJointImpedanceStiffness())),
        damping(damping.has_value() ? *damping : defaultJointImpedanceDamping(this->stiffness)) {
    validate();
  }

  /** Joint stiffness gains [Nm/rad]. */
  Vector7d stiffness{defaultJointImpedanceStiffness()};

  /** Joint damping gains [Nms/rad]. */
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

  /** Maximum absolute joint position error [rad] used by the joint-space controller. */
  Vector7d error_clip{Vector7d::Constant(0.5)};

  /** Constant torque offset added to every command in [Nm]. */
  Vector7d constant_torque_offset{Vector7d::Zero()};

  /** Compensate Coriolis forces using the robot model. */
  bool compensate_coriolis{true};

  /** Shared torque safety limits and soft joint-limit repulsion settings. */
  TorqueSafetyParams safety{};

  /** Joint friction compensation settings. */
  FrictionCompensationParams friction{};

  /** Optional Cartesian-space stiffness/damping projected through the current Jacobian. */
  std::optional<CartesianImpedanceGains> cartesian_gains{std::nullopt};

  /** @brief Throw std::invalid_argument if any parameter is out of range. */
  void validate() const {
    validateNonNegativeFinite(stiffness, "stiffness");
    validateNonNegativeFinite(damping, "damping");
    validateNonNegativeFinite(error_clip, "error_clip");
    if (cartesian_gains.has_value()) {
      cartesian_gains->validate();
    }
    safety.validate();
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
  /**
   * @brief The target joint positions of the motion [rad].
   */
  [[nodiscard]] const Vector7d &target() const { return target_; }

  /**
   * @brief The target joint velocities of the motion [rad/s].
   */
  [[nodiscard]] const Vector7d &target_velocity() const { return target_velocity_; }

  /**
   * @brief The parameters of the motion.
   */
  [[nodiscard]] const JointImpedanceParams &params() const { return params_; }

  /**
   * @brief Set the target impedance gains.
   *
   * The gains are validated and then smoothed in the control loop via
   * exponential interpolation.
   * @param gains The new target gains.
   */
  void setGains(const JointImpedanceGains &gains) {
    gains.validate();
    gains_handle_.set(gains);
  }

  /**
   * @brief Get a copy of the current target impedance gains.
   *
   * Mutating the returned object has no effect on the motion; pass it to
   * setGains to apply changes.
   */
  [[nodiscard]] JointImpedanceGains getGains() const { return gains_handle_.getLastWritten(); }

  /**
   * @brief Set the target Cartesian shaping gains.
   *
   * The gains are validated and then smoothed in the control loop via
   * exponential interpolation.
   * @param gains The new target Cartesian gains.
   */
  void setCartesianGains(const CartesianImpedanceGains &gains) {
    gains.validate();
    cartesian_gains_handle_.set(gains);
  }

  /**
   * @brief Get a copy of the current target Cartesian shaping gains.
   *
   * Mutating the returned object has no effect on the motion; pass it to
   * setCartesianGains to apply changes.
   */
  [[nodiscard]] CartesianImpedanceGains getCartesianGains() const { return cartesian_gains_handle_.getLastWritten(); }

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
  struct CartesianShapingState {
    Matrix6d stiffness;
    Matrix6d damping;                                    // always concrete, interpolated
    Matrix6d critical_damping;                           // cached critical(stiffness)
    std::optional<Matrix6d> critical_damping_stiffness;  // stiffness the cache was computed for
  };

  // Critical damping for the shaping stiffness; recomputes the eigendecomposition only while the
  // stiffness moves and caches it otherwise.
  static const Matrix6d &criticalShapingDamping(CartesianShapingState &shaping);

  WaitFreeTripleBuffer<JointImpedanceGains> gains_handle_;
  WaitFreeTripleBuffer<CartesianImpedanceGains> cartesian_gains_handle_;
  double gains_time_constant_;
  Vector7d current_stiffness_;
  Vector7d current_damping_;
  std::optional<CartesianShapingState> cartesian_shaping_;
};

}  // namespace franky
