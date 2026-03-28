#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <optional>

#include "franky/motion/cartesian_impedance_base.hpp"
#include "franky/motion/reference_type.hpp"
#include "franky/robot_pose.hpp"

namespace franky {

/**
 * @brief Exponential cartesian impedance motion.
 *
 * This motion is a implements a exponential cartesian impedance controller on
 * the client side and does not use Franka's internal impedance controller.
 * Instead, it uses Franka's internal torque controller and calculates the
 * torques itself.
 */
class ExponentialImpedanceMotion : public CartesianImpedanceBase {
 public:
  /**
   * @brief Parameters for the exponential cartesian impedance motion.
   * @see CartesianImpedanceBase::Params
   */
  struct Params : public CartesianImpedanceBase::Params {
    /** The type of the target reference (relative or absolute). */
    ReferenceType target_type{ReferenceType::kAbsolute};

    /** The exponential decay factor for the impedance controller. */
    double exponential_decay{0.005};
  };

  /**
   * @param target The target pose.
   */
  explicit ExponentialImpedanceMotion(const Affine &target);

  /**
   * @param target The target pose.
   * @param params Parameters for the motion.
   */
  explicit ExponentialImpedanceMotion(const Affine &target, const Params &params);

 protected:
  void initImpl(const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) override;

  std::tuple<CartesianReference, bool> update(
      const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time,
      franka::Duration abs_time) override;

 private:
  Params params_;
};

}  // namespace franky
