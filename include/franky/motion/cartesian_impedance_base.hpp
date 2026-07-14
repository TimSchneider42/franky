#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "franky/motion/motion.hpp"
#include "franky/motion/torque_control_utils.hpp"
#include "franky/twist.hpp"
#include "franky/twist_acceleration.hpp"
#include "franky/wait_free_triple_buffer.hpp"

namespace franky {

/**
 * @brief Cartesian impedance reference expressed in the base frame.
 */
struct CartesianReference {
  /** Desired end-effector pose. */
  Affine target{Affine::Identity()};

  /**
   * Desired end-effector twist in the base frame.
   *
   * When present, the damping term acts on twist error rather than resisting
   * all motion toward zero.
   */
  std::optional<Twist> target_twist{};

  /**
   * Desired end-effector acceleration in the base frame.
   *
   * When present, the controller adds a model-based inertial feedforward
   * wrench Lambda(q) * target_acceleration before mapping through J^T.
   */
  std::optional<TwistAcceleration> target_acceleration{};

  /** @brief Throw std::invalid_argument if any value is non-finite. */
  void validate() const {
    validateFinite(target.matrix(), "target");
    if (target_twist.has_value()) validateFinite(target_twist->vector_repr(), "target_twist");
    if (target_acceleration.has_value()) validateFinite(target_acceleration->vector_repr(), "target_acceleration");
  }
};

inline Matrix6d cartesianGainBlocks(double translational, double rotational) {
  Matrix6d gains = Matrix6d::Zero();
  gains.topLeftCorner<3, 3>() = translational * Eigen::Matrix3d::Identity();
  gains.bottomRightCorner<3, 3>() = rotational * Eigen::Matrix3d::Identity();
  return gains;
}

inline Matrix6d defaultCartesianImpedanceStiffness() { return cartesianGainBlocks(500.0, 50.0); }

inline Matrix6d defaultCartesianImpedanceDamping(const Matrix6d &stiffness) {
  Eigen::SelfAdjointEigenSolver<Matrix6d> solver(stiffness);
  return 2.0 * solver.operatorSqrt();
}

/**
 * @brief Runtime-adjustable stiffness and damping gains for Cartesian
 * impedance motions.
 */
struct CartesianImpedanceGains {
  CartesianImpedanceGains() = default;

  explicit CartesianImpedanceGains(Matrix6d stiffness, std::optional<Matrix6d> damping = std::nullopt)
      : stiffness(std::move(stiffness)), damping(std::move(damping)) {
    validate();
  }

  /**
   * @brief Create gains with uniform translational and rotational components.
   *
   * @param translational_stiffness Stiffness applied to all translational axes [N/m].
   * @param rotational_stiffness    Stiffness applied to all rotational axes [Nm/rad].
   * @param translational_damping   Damping applied to all translational axes
   * [Ns/m]. If unset, critical damping is used.
   * @param rotational_damping      Damping applied to all rotational axes
   * [Nms/rad]. If unset, critical damping is used.
   */
  static CartesianImpedanceGains isotropic(
      double translational_stiffness, double rotational_stiffness,
      std::optional<double> translational_damping = std::nullopt,
      std::optional<double> rotational_damping = std::nullopt) {
    CartesianImpedanceGains gains;
    gains.stiffness = cartesianGainBlocks(translational_stiffness, rotational_stiffness);
    if (translational_damping.has_value() || rotational_damping.has_value()) {
      gains.damping = cartesianGainBlocks(
          translational_damping.value_or(2.0 * std::sqrt(translational_stiffness)),
          rotational_damping.value_or(2.0 * std::sqrt(rotational_stiffness)));
    }
    gains.validate();
    return gains;
  }

  /**
   * @brief Create gains from per-axis diagonal entries.
   *
   * @param stiffness Per-axis stiffness [N/m, Nm/rad], ordered [x, y, z, rx, ry, rz].
   * @param damping   Per-axis damping [Ns/m, Nms/rad], ordered [x, y, z, rx,
   * ry, rz]. If unset, critical damping is used.
   */
  static CartesianImpedanceGains diagonal(const Vector6d &stiffness, std::optional<Vector6d> damping = std::nullopt) {
    CartesianImpedanceGains gains;
    gains.stiffness = stiffness.asDiagonal();
    if (damping.has_value()) gains.damping = damping->asDiagonal();
    gains.validate();
    return gains;
  }

  /** Cartesian stiffness matrix [N/m, Nm/rad], ordered [x, y, z, rx, ry, rz]. */
  Matrix6d stiffness{defaultCartesianImpedanceStiffness()};

  /** Cartesian damping matrix. If unset, critical damping is used. */
  std::optional<Matrix6d> damping{std::nullopt};

  /** @brief Throw std::invalid_argument unless gains are symmetric positive semidefinite matrices. */
  void validate() const {
    validatePositiveSemidefinite(stiffness, "stiffness");
    if (damping.has_value()) validatePositiveSemidefinite(*damping, "damping");
  }
};

/**
 * @brief Joint-posture objective projected into the Cartesian nullspace.
 */
struct PostureTask {
  PostureTask() = default;

  PostureTask(
      const Vector7d &target, const Vector7d &stiffness, std::optional<Vector7d> damping = std::nullopt,
      std::optional<double> max_torque = std::nullopt)
      : target(target), stiffness(stiffness), damping(std::move(damping)), max_torque(max_torque) {
    validate();
  }

  /** Convenience constructor applying the same scalar gains to all joints. */
  PostureTask(
      const Vector7d &target, double stiffness, std::optional<double> damping = std::nullopt,
      std::optional<double> max_torque = std::nullopt)
      : target(target), stiffness(Vector7d::Constant(stiffness)), max_torque(max_torque) {
    if (damping.has_value()) this->damping = Vector7d::Constant(*damping);
    validate();
  }

  /** Preferred joint posture [rad]. */
  Vector7d target{Vector7d::Zero()};

  /**
   * Per-joint posture stiffness in [Nm/rad].
   *
   * A joint with zero stiffness is not pushed by this task. Note that the
   * task torque is projected into the Jacobian nullspace, so pushing a single
   * joint still moves all joints that participate in the self-motion.
   */
  Vector7d stiffness{Vector7d::Zero()};

  /**
   * Per-joint posture damping in [Nms/rad].
   *
   * If unset, the controller uses critical damping, 2*sqrt(stiffness), per joint.
   */
  std::optional<Vector7d> damping{std::nullopt};

  /** Per-joint absolute torque clamp for this task [Nm]. Unset means no clamp. */
  std::optional<double> max_torque{std::nullopt};

  /** @brief Throw std::invalid_argument if any task parameter is out of range. */
  void validate() const {
    validateFinite(target, "nullspace posture target");
    validateNonNegativeFinite(stiffness, "nullspace posture stiffness");
    if (damping.has_value()) validateNonNegativeFinite(*damping, "nullspace posture damping");
    if (max_torque.has_value()) validateNonNegativeFinite(*max_torque, "nullspace posture max_torque");
  }
};

/**
 * @brief Manipulability maximization objective projected into the Cartesian nullspace.
 */
struct ManipulabilityTask {
  ManipulabilityTask() = default;

  ManipulabilityTask(double gain, double damping = 0.0, std::optional<double> max_torque = std::nullopt)
      : gain(gain), damping(damping), max_torque(max_torque) {
    validate();
  }

  /** Gain applied to the manipulability gradient. */
  double gain{0.0};

  /** Joint damping applied to this task before projection [Nms/rad]. */
  double damping{0.0};

  /** Per-joint absolute torque clamp for this task [Nm]. Unset means no clamp. */
  std::optional<double> max_torque{std::nullopt};

  /** @brief Throw std::invalid_argument if any task parameter is out of range. */
  void validate() const {
    validateNonNegativeFinite(gain, "nullspace manipulability gain");
    validateNonNegativeFinite(damping, "nullspace manipulability damping");
    if (max_torque.has_value()) validateNonNegativeFinite(*max_torque, "nullspace manipulability max_torque");
  }
};

using NullspaceTask = std::variant<PostureTask, ManipulabilityTask>;

/**
 * @brief Runtime-adjustable gains for a nullspace task.
 */
struct NullspaceGains {
  /** Per-joint posture stiffness [Nm/rad]. A joint with zero stiffness is not pushed. */
  Vector7d posture_stiffness{Vector7d::Zero()};

  /** Per-joint posture damping [Nms/rad]. If unset, critical damping is used. */
  std::optional<Vector7d> posture_damping{std::nullopt};

  /** Absolute torque clamp for the posture task [Nm]. Unset means no clamp. */
  std::optional<double> posture_max_torque{std::nullopt};

  /** Gain applied to the manipulability gradient. */
  double manipulability_gain{0.0};

  /** Joint damping applied to the manipulability task before projection [Nms/rad]. */
  double manipulability_damping{0.0};

  /** Absolute torque clamp for the manipulability task [Nm]. Unset means no clamp. */
  std::optional<double> manipulability_max_torque{std::nullopt};

  /** @brief Throw std::invalid_argument if any gain or clamp is out of range. */
  void validate() const {
    validateNonNegativeFinite(posture_stiffness, "nullspace posture stiffness");
    if (posture_damping.has_value()) validateNonNegativeFinite(*posture_damping, "nullspace posture damping");
    if (posture_max_torque.has_value()) validateNonNegativeFinite(*posture_max_torque, "nullspace posture max_torque");
    validateNonNegativeFinite(manipulability_gain, "nullspace manipulability gain");
    validateNonNegativeFinite(manipulability_damping, "nullspace manipulability damping");
    if (manipulability_max_torque.has_value())
      validateNonNegativeFinite(*manipulability_max_torque, "nullspace manipulability max_torque");
  }
};

/**
 * @brief Base class for client-side cartesian impedance motions.
 *
 * This motion implements a cartesian impedance controller on the client
 * side and does not use Franka's internal impedance controller. Instead, it
 * uses Franka's internal torque controller and calculates the torques itself.
 */
class CartesianImpedanceBase : public Motion<franka::Torques> {
 public:
  /**
   * @brief Parameters for the impedance motion.
   */
  struct Params {
    /** Cartesian stiffness matrix [N/m, Nm/rad], ordered [x, y, z, rx, ry, rz]. */
    Matrix6d stiffness{defaultCartesianImpedanceStiffness()};

    /** Cartesian damping matrix. If unset, critical damping is used. */
    std::optional<Matrix6d> damping{std::nullopt};

    /**
     * Maximum absolute Cartesian position error [m] used by the task-space controller.
     *
     * The translational error is clamped elementwise before the impedance wrench
     * is computed. This bounds the commanded Cartesian force when the reference
     * jumps or contact prevents the end effector from reaching the target.
     */
    Eigen::Vector3d translational_error_clip{Eigen::Vector3d::Constant(0.10)};

    /**
     * Maximum absolute Cartesian orientation error [rad] used by the task-space controller.
     *
     * The rotational error is clamped elementwise in the base frame before the
     * impedance wrench is computed. This bounds the commanded Cartesian torque.
     */
    Eigen::Vector3d rotational_error_clip{Eigen::Vector3d::Constant(0.25)};

    /** Per-axis force/torque constraints [N, Nm]. nullopt on an axis means unconstrained. */
    std::array<std::optional<double>, 6> force_constraints{};

    /**
     * Nullspace objectives.
     *
     * Each task contributes a joint-space torque that is summed and projected
     * into the Jacobian nullspace. At most one task of each concrete type is
     * supported because runtime-adjustable gains are shared by task type.
     */
    std::vector<NullspaceTask> nullspace_tasks{};

    /** Shared torque safety limits and soft joint-limit repulsion settings. */
    TorqueSafetyParams safety{};

    /** Per-joint friction feedforward. Defaults to zero (disabled). */
    FrictionCompensationParams friction{};

    /** @brief Throw std::invalid_argument if any parameter is out of range. */
    void validate() const {
      validatePositiveSemidefinite(stiffness, "stiffness");
      if (damping.has_value()) validatePositiveSemidefinite(*damping, "damping");
      validateNonNegativeFinite(translational_error_clip, "translational_error_clip");
      validateNonNegativeFinite(rotational_error_clip, "rotational_error_clip");
      bool has_posture_task = false;
      bool has_manipulability_task = false;
      for (const auto &task : nullspace_tasks) {
        std::visit([](const auto &concrete_task) { concrete_task.validate(); }, task);
        if (std::holds_alternative<PostureTask>(task)) {
          if (has_posture_task) throw std::invalid_argument("Only one PostureTask is supported.");
          has_posture_task = true;
        } else {
          if (has_manipulability_task) throw std::invalid_argument("Only one ManipulabilityTask is supported.");
          has_manipulability_task = true;
        }
      }
      safety.validate();
      friction.validate();
    }
  };

  /**
   * @brief The target pose of the motion.
   */
  [[nodiscard]] const Affine &target() const { return target_; }

  /**
   * @brief Set the target impedance gains.
   *
   * The gains are validated and then smoothed in the control loop via
   * exponential interpolation.
   * @param gains The new target gains.
   */
  void setGains(const CartesianImpedanceGains &gains) {
    gains.validate();
    gains_handle_.set(gains);
  }

  /**
   * @brief Get a copy of the current target impedance gains.
   *
   * Mutating the returned object has no effect on the motion; pass it to
   * setGains to apply changes.
   */
  [[nodiscard]] CartesianImpedanceGains getGains() const { return gains_handle_.getLastWritten(); }

  /**
   * @brief Set the target nullspace gains.
   *
   * The gains are smoothed in the control loop via exponential interpolation.
   * @param gains The new target nullspace gains.
   */
  void setNullspaceGains(const NullspaceGains &gains) {
    gains.validate();
    nullspace_gains_handle_.set(gains);
  }

  /**
   * @brief Get a copy of the current target nullspace gains.
   *
   * Mutating the returned object has no effect on the motion; pass it to
   * setNullspaceGains to apply changes.
   */
  [[nodiscard]] NullspaceGains getNullspaceGains() const { return nullspace_gains_handle_.getLastWritten(); }

 protected:
  /**
   * @param target The target pose.
   * @param params Parameters for the motion.
   * @param gains_time_constant Smoothing time constant for gain transitions [s].
   *        Default 0.1s.
   */
  explicit CartesianImpedanceBase(Affine target, const Params &params, double gains_time_constant = 0.1);

  [[nodiscard]] franka::Torques computeCommand(
      const RobotState &robot_state, const CartesianReference &reference, double dt);

  [[nodiscard]] inline const Params &base_params() const { return params_; }

  Affine target_;

 private:
  const Matrix6d &criticalDamping();

  Params params_;

  WaitFreeTripleBuffer<CartesianImpedanceGains> gains_handle_;
  WaitFreeTripleBuffer<NullspaceGains> nullspace_gains_handle_;
  double gains_time_constant_;
  Matrix6d current_stiffness_;
  Matrix6d current_damping_;
  NullspaceGains current_nullspace_gains_;

  /** Cached critical damping = defaultCartesianImpedanceDamping(current_stiffness_). */
  Matrix6d critical_damping_;

  /**
   * Stiffness for which critical_damping_ was last computed. Lets criticalDamping() skip the
   * eigendecomposition while the stiffness is unchanged.
   */
  std::optional<Matrix6d> critical_damping_stiffness_;
};

using ImpedanceMotion = CartesianImpedanceBase;

}  // namespace franky
