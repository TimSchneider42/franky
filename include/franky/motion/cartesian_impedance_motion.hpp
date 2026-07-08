#pragma once

#include <optional>

#include "franky/motion/cartesian_impedance_base.hpp"
#include "franky/motion/reference_type.hpp"
#include "franky/twist.hpp"

namespace franky {

/**
 * @brief Cartesian impedance motion.
 *
 * This motion implements a cartesian impedance controller on the client side
 * and does not use Franka's internal impedance controller. Instead, it uses
 * Franka's internal torque controller and calculates the torques itself.
 *
 * Analogous to JointImpedanceMotion, this motion regulates toward a fixed
 * target pose (and optional target twist) and does not terminate on its own.
 * It runs until it is preempted or stopped.
 */
class CartesianImpedanceMotion : public CartesianImpedanceBase {
 public:
  /**
   * @brief Parameters for the Cartesian impedance motion.
   * @see CartesianImpedanceBase::Params
   */
  struct Params : public CartesianImpedanceBase::Params {
    /** The type of the target reference (relative or absolute). */
    ReferenceType target_type{ReferenceType::kAbsolute};
  };

  /**
   * @param target The target pose.
   */
  explicit CartesianImpedanceMotion(const Affine &target);

  /**
   * @param target The target pose.
   * @param params Parameters for the motion.
   */
  explicit CartesianImpedanceMotion(const Affine &target, const Params &params);

  /**
   * @param target The target pose.
   * @param target_twist The target twist in the base frame. The damping term
   * acts on twist error rather than resisting all motion toward zero.
   */
  CartesianImpedanceMotion(const Affine &target, const Twist &target_twist);

  /**
   * @param target The target pose.
   * @param target_twist The target twist in the base frame. The damping term
   * acts on twist error rather than resisting all motion toward zero.
   * @param params Parameters for the motion.
   */
  CartesianImpedanceMotion(const Affine &target, const Twist &target_twist, const Params &params);

  [[nodiscard]] const Twist &target_twist() const { return target_twist_; }
  [[nodiscard]] const Params &params() const { return params_; }

 protected:
  void initImpl(const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) override;

  franka::Torques nextCommandImpl(
      const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
      const std::optional<franka::Torques> &previous_command) override;

 private:
  /** The target pose as passed to the constructor, before any relative-target
   * resolution at motion start. */
  Affine original_target_;
  Twist target_twist_;
  Params params_;
};

}  // namespace franky
