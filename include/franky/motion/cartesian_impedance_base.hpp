#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <map>
#include <memory>
#include <optional>

#include "franky/motion/impedance_gains_handle.hpp"
#include "franky/motion/joint_impedance_motion.hpp"
#include "franky/motion/motion.hpp"
#include "franky/robot_pose.hpp"
#include "franky/twist.hpp"

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
};

/**
 * @brief Base class for client-side cartesian impedance motions.
 *
 * This motion is implements a cartesian impedance controller on the client
 * side and does not use Franka's internal impedance controller. Instead, it
 * uses Franka's internal torque controller and calculates the torques itself.
 */
class CartesianImpedanceBase : public Motion<franka::Torques> {
 public:
  /**
   * @brief Parameters for the impedance motion.
   */
  struct Params : public TorqueSafetyParams {
    /** The translational stiffness in [10, 3000] N/m. */
    double translational_stiffness{2000};

    /** The rotational stiffness in [1, 300] Nm/rad. */
    double rotational_stiffness{200};

    /** The force constraints in [N, Nm] for each joint. */
    Eigen::Vector<double, 6> force_constraints;

    /** Allows to enable or disable individual force constraints. */
    Eigen::Vector<bool, 6> force_constraints_active{Eigen::Vector<bool, 6>::Zero()};

    /**
     * Preferred joint posture for the Cartesian controller nullspace.
     *
     * When set together with a positive nullspace_stiffness, the controller
     * adds a secondary posture torque that does not change the Cartesian task
     * to first order.
     */
    std::optional<Vector7d> nullspace_target{std::nullopt};

    /** Nullspace posture stiffness in [Nm/rad]. Set to 0 to disable. */
    double nullspace_stiffness{0.0};
  };

  /**
   * @param target The target pose.
   * @param params Parameters for the motion.
   * @param gains_handle Optional handle for runtime gain updates. When null,
   *        gains are static (set once from params). When present, gains are
   *        read each cycle and exponentially interpolated toward the target.
   * @param gains_time_constant Smoothing time constant for gain transitions [s].
   *        Only used when gains_handle is provided. Default 0.1s.
   */
  explicit CartesianImpedanceBase(Affine target, const Params &params,
                                  std::shared_ptr<CartesianImpedanceGainsHandle> gains_handle = nullptr,
                                  double gains_time_constant = 0.1);

 protected:
  void initImpl(const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) override;

  franka::Torques nextCommandImpl(
      const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
      const std::optional<franka::Torques> &previous_command) override;

  [[nodiscard]] inline Affine intermediate_target() const { return intermediate_target_; }

  [[nodiscard]] inline Affine target() const { return absolute_target_; }

  inline void setAbsoluteTarget(const Affine &target) {
    target_ = target;
    absolute_target_ = target;
  }

  virtual std::tuple<CartesianReference, bool> update(
      const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time) = 0;

 private:
  void rebuildStiffnessDamping();

  Affine absolute_target_;
  Affine target_;
  Params params_;

  std::shared_ptr<CartesianImpedanceGainsHandle> gains_handle_;
  double gains_time_constant_;
  double current_translational_stiffness_;
  double current_rotational_stiffness_;
  double current_nullspace_stiffness_;

  Eigen::Matrix<double, 6, 6> stiffness, damping;
  Affine intermediate_target_;

  std::unique_ptr<franka::Model> model_;
};

using ImpedanceMotion = CartesianImpedanceBase;

}  // namespace franky
