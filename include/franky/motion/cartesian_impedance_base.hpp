#pragma once

#include <Eigen/Core>
#include <array>
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

struct CartesianImpedanceGains {
  CartesianImpedanceGains() = default;

  explicit CartesianImpedanceGains(
      double translational_stiffness, double rotational_stiffness, double nullspace_stiffness = 0.0)
      : translational_stiffness(translational_stiffness),
        rotational_stiffness(rotational_stiffness),
        nullspace_stiffness(nullspace_stiffness) {
    validate();
  }

  double translational_stiffness{500.0};
  double rotational_stiffness{50.0};
  double nullspace_stiffness{0.0};

  /** @brief Throw std::invalid_argument if any gain is negative or non-finite. */
  void validate() const {
    validateNonNegativeFinite(translational_stiffness, "translational_stiffness");
    validateNonNegativeFinite(rotational_stiffness, "rotational_stiffness");
    validateNonNegativeFinite(nullspace_stiffness, "nullspace_stiffness");
  }
};

/**
 * @brief Joint-posture objective projected into the Cartesian nullspace.
 */
struct PostureTask {
  PostureTask() = default;

  PostureTask(
      const Vector7d &target, double stiffness, std::optional<double> damping = std::nullopt, double max_torque = 0.0)
      : target(target), stiffness(stiffness), damping(damping), max_torque(max_torque) {}

  /** Preferred joint posture [rad]. */
  Vector7d target{Vector7d::Zero()};

  /** Posture stiffness in [Nm/rad]. */
  double stiffness{0.0};

  /**
   * Posture damping in [Nms/rad].
   *
   * If unset, the controller uses critical damping, 2*sqrt(stiffness).
   */
  std::optional<double> damping{std::nullopt};

  /** Per-joint absolute torque clamp for this task [Nm]. Set <= 0 to disable. */
  double max_torque{0.0};
};

/**
 * @brief Manipulability maximization objective projected into the Cartesian nullspace.
 */
struct ManipulabilityTask {
  ManipulabilityTask() = default;

  ManipulabilityTask(double gain, double damping = 0.0, double max_torque = 0.0)
      : gain(gain), damping(damping), max_torque(max_torque) {}

  /** Gain applied to the manipulability gradient. */
  double gain{0.0};

  /** Joint damping applied to this task before projection [Nms/rad]. */
  double damping{0.0};

  /** Per-joint absolute torque clamp for this task [Nm]. Set <= 0 to disable. */
  double max_torque{0.0};
};

using NullspaceTask = std::variant<PostureTask, ManipulabilityTask>;

/**
 * @brief Base class for client-side cartesian impedance motions.
 *
 * This class computes joint torques from a task-space spring-damper law with
 * optional nullspace posture control and model compensation. It does not use
 * Franka's internal impedance controller. Instead, it uses Franka's internal
 * torque controller and calculates the torques itself. Subclasses implement
 * nextCommandImpl and call computeCommand with their current reference.
 */
class CartesianImpedanceBase : public Motion<franka::Torques> {
 public:
  /**
   * @brief Parameters for the impedance motion.
   */
  struct Params {
    /** The translational stiffness in [10, 3000] N/m. */
    double translational_stiffness{500};

    /** The rotational stiffness in [1, 300] Nm/rad. */
    double rotational_stiffness{50};

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
     * into the Jacobian nullspace.
     */
    std::vector<NullspaceTask> nullspace_tasks{};

    /** Shared torque safety limits and soft joint-limit repulsion settings. */
    TorqueSafetyParams safety{};

    /** Per-joint friction feedforward. Defaults to zero (disabled). */
    FrictionCompensationParams friction{};

    /** @brief Throw std::invalid_argument if any parameter is out of range. */
    void validate() const {
      validateNonNegativeFinite(translational_stiffness, "translational_stiffness");
      validateNonNegativeFinite(rotational_stiffness, "rotational_stiffness");
      friction.validate();
    }
  };

  [[nodiscard]] const Affine &target() const { return target_; }

  void setGains(const CartesianImpedanceGains &gains) {
    gains.validate();
    gains_handle_.set(gains);
  }
  [[nodiscard]] CartesianImpedanceGains getGains() const { return gains_handle_.getLastWritten(); }

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
  void rebuildStiffnessDamping();

  Params params_;

  WaitFreeTripleBuffer<CartesianImpedanceGains> gains_handle_;
  double gains_time_constant_;
  double current_translational_stiffness_;
  double current_rotational_stiffness_;

  Eigen::Matrix<double, 6, 6> stiffness, damping;
};

using ImpedanceMotion = CartesianImpedanceBase;

}  // namespace franky
