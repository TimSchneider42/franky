#include "franky/motion/cartesian_impedance_motion.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>

#include "franky/motion/impedance_motion.hpp"

namespace franky {

CartesianImpedanceMotion::CartesianImpedanceMotion(const Affine &target, franka::Duration duration)
    : CartesianImpedanceMotion(target, duration, Params()) {}

CartesianImpedanceMotion::CartesianImpedanceMotion(
    const Affine &target, franka::Duration duration, const CartesianImpedanceMotion::Params &params)
    : duration_(duration), params_(params), ImpedanceMotion(target, params) {}

void CartesianImpedanceMotion::initImpl(
    const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) {
  ImpedanceMotion::initImpl(robot_state, previous_command);
  initial_pose_ = Affine(Eigen::Matrix4d::Map(robot_state.O_T_EE_c.data()));
}

std::tuple<Affine, bool> CartesianImpedanceMotion::update(
    const RobotState &robot_state, franka::Duration time_step, franka::Duration time) {
  double transition_parameter = time / duration_;
  Affine intermediate_goal;
  bool done;
  if (transition_parameter <= 1.0) {  // [ms] to [s]
    Eigen::Quaterniond q_start(initial_pose_.rotation());
    Eigen::Quaterniond q_end(target().rotation());
    auto init_trans = initial_pose_.translation();
    auto trans = init_trans + transition_parameter * (target().translation() - init_trans);
    auto rot = q_start.slerp(transition_parameter, q_end);
    intermediate_goal.fromPositionOrientationScale(trans, rot, Eigen::Vector3d::Ones());
    done = false;
  } else if (params_.return_when_finished && transition_parameter > params_.finish_wait_factor) {
    done = true;
    intermediate_goal = target();
  }
  return {intermediate_goal, done};
}

}  // namespace franky
