#include "franky/motion/cartesian_impedance_motion.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace franky {

CartesianImpedanceMotion::CartesianImpedanceMotion(const Affine &target) : CartesianImpedanceMotion(target, Params()) {}

CartesianImpedanceMotion::CartesianImpedanceMotion(const Affine &target, const Params &params)
    : CartesianImpedanceMotion(target, Twist(), params) {}

CartesianImpedanceMotion::CartesianImpedanceMotion(const Affine &target, const Twist &target_twist)
    : CartesianImpedanceMotion(target, target_twist, Params()) {}

CartesianImpedanceMotion::CartesianImpedanceMotion(
    const Affine &target, const Twist &target_twist, const CartesianImpedanceMotion::Params &params)
    : CartesianImpedanceBase(target, params), original_target_(target), target_twist_(target_twist), params_(params) {}

void CartesianImpedanceMotion::initImpl(
    const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) {
  if (params_.target_type == ReferenceType::kRelative) {
    target_ = Affine(Eigen::Matrix4d::Map(robot_state.O_T_EE.data())) * original_target_;
  } else {
    target_ = original_target_;
  }
}

franka::Torques CartesianImpedanceMotion::nextCommandImpl(
    const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
    const std::optional<franka::Torques> &previous_command) {
  CartesianReference reference;
  reference.target = target_;
  reference.target_twist = target_twist_;
  const double dt = time_step.toSec();
  return computeCommand(robot_state, reference, dt);
}

}  // namespace franky
