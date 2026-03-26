#include "franky/motion/exponential_impedance_motion.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>

#include "franky/motion/cartesian_impedance_base.hpp"

namespace franky {

ExponentialImpedanceMotion::ExponentialImpedanceMotion(const Affine &target)
    : ExponentialImpedanceMotion(target, Params()) {}

ExponentialImpedanceMotion::ExponentialImpedanceMotion(
    const Affine &target, const ExponentialImpedanceMotion::Params &params)
    : params_(params), CartesianImpedanceBase(target, params) {}

std::tuple<CartesianReference, bool> ExponentialImpedanceMotion::update(
    const RobotState &robot_state, franka::Duration time_step, franka::Duration time, franka::Duration /*abs_time*/) {
  auto trans = params_.exponential_decay * target().translation() +
               (1.0 - params_.exponential_decay) * intermediate_target().translation();
  auto rot = Eigen::Quaterniond(intermediate_target().rotation())
                 .slerp(params_.exponential_decay, Eigen::Quaterniond(target().rotation()));
  return {CartesianReference{Affine().fromPositionOrientationScale(trans, rot, Eigen::Vector3d::Ones()), std::nullopt},
          false};
}

}  // namespace franky
