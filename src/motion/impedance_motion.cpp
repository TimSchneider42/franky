#include "franky/motion/impedance_motion.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <utility>

#include "franky/motion/motion.hpp"
#include "franky/robot_pose.hpp"

namespace franky {

ImpedanceMotion::ImpedanceMotion(Affine target, const ImpedanceMotion::Params &params)
    : target_(std::move(target)), params_(params), Motion<franka::Torques>() {
  stiffness.setZero();
  stiffness.topLeftCorner(3, 3) << params.translational_stiffness * Eigen::MatrixXd::Identity(3, 3);
  stiffness.bottomRightCorner(3, 3) << params.rotational_stiffness * Eigen::MatrixXd::Identity(3, 3);
  damping.setZero();
  damping.topLeftCorner(3, 3) << 2.0 * sqrt(params.translational_stiffness) * Eigen::MatrixXd::Identity(3, 3);
  damping.bottomRightCorner(3, 3) << 2.0 * sqrt(params.rotational_stiffness) * Eigen::MatrixXd::Identity(3, 3);
}

void ImpedanceMotion::initImpl(const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) {
  auto robot_pose = Affine(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  intermediate_target_ = robot_pose;
  if (params_.target_type == ReferenceType::kRelative)
    absolute_target_ = robot_pose * target_;
  else
    absolute_target_ = target_;
}

franka::Torques ImpedanceMotion::nextCommandImpl(
    const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
    const std::optional<franka::Torques> &previous_command) {
  auto model = robot()->model();
  Vector7d coriolis = model->coriolis(robot_state);
  Jacobian jacobian = model->zeroJacobian(franka::Frame::kEndEffector, robot_state);

  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::Quaterniond orientation(transform.rotation());

  Eigen::Matrix<double, 6, 1> error;
  error.head(3) << robot_state.O_T_EE.translation() - intermediate_target_.translation();

  Eigen::Quaterniond quat(target().rotation());
  if (quat.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }

  Eigen::Quaterniond error_quaternion(orientation.inverse() * quat);
  error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  error.tail(3) << -transform.linear() * error.tail(3);

  auto wrench_cartesian_default = -stiffness * error - damping * (jacobian * robot_state.dq);
  auto wrench_cartesian = params_.force_constraints_active.select(params_.force_constraints, wrench_cartesian_default);

  auto tau_task = jacobian.transpose() * wrench_cartesian;
  auto tau_d = tau_task + coriolis;

  std::array<double, 7> tau_d_array{};
  Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_d;

  // Update target with target motion
  auto [intermediate_target, finish] = update(robot_state, time_step, rel_time);
  intermediate_target_ = intermediate_target;

  auto output = franka::Torques(tau_d_array);
  if (finish) output = franka::MotionFinished(output);

  return output;
}

}  // namespace franky
