#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <optional>

#include "franky/motion/impedance_motion.hpp"
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
class ExponentialImpedanceMotion : public ImpedanceMotion {
 public:
  /**
   * @brief Parameters for the exponential cartesian impedance motion.
   * @see ImpedanceMotion::Params
   */
  struct Params : public ImpedanceMotion::Params {
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
  std::tuple<Affine, bool> update(
      const RobotState &robot_state, franka::Duration time_step, franka::Duration time) override;

 private:
  Params params_;
};

}  // namespace franky
