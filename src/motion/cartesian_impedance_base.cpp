#include "franky/motion/cartesian_impedance_base.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <map>
#include <utility>

#include "franky/motion/motion.hpp"
#include "franky/motion/torque_control_utils.hpp"
#include "franky/robot_pose.hpp"

namespace franky {

namespace {
Eigen::Matrix<double, 6, 7> pseudoInverse(const Eigen::Matrix<double, 7, 6> &matrix) {
  Eigen::JacobiSVD<Eigen::Matrix<double, 7, 6>> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const auto &singular_values = svd.singularValues();
  Eigen::Matrix<double, 6, 7> sigma_pinv = Eigen::Matrix<double, 6, 7>::Zero();
  constexpr double tolerance = 1e-6;
  for (int i = 0; i < singular_values.size(); ++i) {
    if (singular_values[i] > tolerance) sigma_pinv(i, i) = 1.0 / singular_values[i];
  }
  return svd.matrixV() * sigma_pinv * svd.matrixU().transpose();
}
}  // namespace

CartesianImpedanceBase::CartesianImpedanceBase(
    Affine target, const CartesianImpedanceBase::Params &params,
    std::shared_ptr<CartesianImpedanceGainsHandle> gains_handle, double gains_time_constant)
    : target_(std::move(target)),
      params_(params),
      gains_handle_(std::move(gains_handle)),
      gains_time_constant_(gains_time_constant),
      current_translational_stiffness_(params.translational_stiffness),
      current_rotational_stiffness_(params.rotational_stiffness),
      current_nullspace_stiffness_(params.nullspace_stiffness),
      Motion<franka::Torques>() {
  rebuildStiffnessDamping();
}

void CartesianImpedanceBase::rebuildStiffnessDamping() {
  stiffness.setZero();
  stiffness.topLeftCorner(3, 3) << current_translational_stiffness_ * Eigen::MatrixXd::Identity(3, 3);
  stiffness.bottomRightCorner(3, 3) << current_rotational_stiffness_ * Eigen::MatrixXd::Identity(3, 3);
  damping.setZero();
  const double translational_damping = 2.0 * std::sqrt(current_translational_stiffness_);
  const double rotational_damping = 2.0 * std::sqrt(current_rotational_stiffness_);
  damping.topLeftCorner(3, 3) << translational_damping * Eigen::MatrixXd::Identity(3, 3);
  damping.bottomRightCorner(3, 3) << rotational_damping * Eigen::MatrixXd::Identity(3, 3);
}

void CartesianImpedanceBase::initImpl(
    const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) {
  auto robot_pose = Affine(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  intermediate_target_ = robot_pose;
  absolute_target_ = target_;
}

franka::Torques CartesianImpedanceBase::nextCommandImpl(
    const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
    const std::optional<franka::Torques> &previous_command) {
  auto [reference, finish] = update(robot_state, time_step, rel_time, abs_time);
  intermediate_target_ = reference.target;

  // If a gains handle is present, interpolate toward the target gains.
  if (gains_handle_ && gains_handle_->hasGains()) {
    const auto target_gains = gains_handle_->get();
    const double dt = time_step.toSec();
    const double alpha = 1.0 - std::exp(-dt / gains_time_constant_);
    current_translational_stiffness_ +=
        alpha * (target_gains.translational_stiffness - current_translational_stiffness_);
    current_rotational_stiffness_ += alpha * (target_gains.rotational_stiffness - current_rotational_stiffness_);
    current_nullspace_stiffness_ += alpha * (target_gains.nullspace_stiffness - current_nullspace_stiffness_);
    rebuildStiffnessDamping();
  }

  auto model = robot()->model();
  Vector7d coriolis = model->coriolis(robot_state);
  Jacobian jacobian = model->zeroJacobian(franka::Frame::kEndEffector, robot_state);

  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::Quaterniond orientation(transform.rotation());

  Eigen::Matrix<double, 6, 1> error;
  error.head(3) << robot_state.O_T_EE.translation() - intermediate_target_.translation();
  error.head(3) = error.head(3).cwiseMax(-params_.translational_error_clip).cwiseMin(params_.translational_error_clip);

  Eigen::Quaterniond quat(intermediate_target_.rotation());
  if (quat.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }

  Eigen::Quaterniond error_quaternion(orientation.inverse() * quat);
  error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  error.tail(3) << -transform.linear() * error.tail(3);
  error.tail(3) = error.tail(3).cwiseMax(-params_.rotational_error_clip).cwiseMin(params_.rotational_error_clip);

  const Vector6d desired_twist =
      reference.target_twist.has_value() ? reference.target_twist->vector_repr() : Vector6d::Zero();
  const Vector6d measured_twist = jacobian * robot_state.dq;

  auto wrench_cartesian_default = -stiffness * error - damping * (measured_twist - desired_twist);
  auto wrench_cartesian = params_.force_constraints_active.select(params_.force_constraints, wrench_cartesian_default);

  auto tau_task = jacobian.transpose() * wrench_cartesian;
  Vector7d tau_nullspace = Vector7d::Zero();
  if (params_.nullspace_target.has_value() && current_nullspace_stiffness_ > 0.0) {
    const auto jacobian_transpose_pinv = pseudoInverse(jacobian.transpose());
    const auto nullspace_projector =
        Eigen::Matrix<double, 7, 7>::Identity() - jacobian.transpose() * jacobian_transpose_pinv;
    const auto nullspace_error = params_.nullspace_target.value() - robot_state.q;
    const double nullspace_damping = 2.0 * std::sqrt(current_nullspace_stiffness_);
    tau_nullspace =
        nullspace_projector * (current_nullspace_stiffness_ * nullspace_error - nullspace_damping * robot_state.dq);
  }

  Vector7d tau_limit = Vector7d::Zero();
  if (params_.joint_limit_repulsion_active) {
    tau_limit = franky::computeJointLimitTorque(
        robot_state.q,
        robot_state.dq,
        params_.lower_joint_limits,
        params_.upper_joint_limits,
        params_.joint_limit_activation_distance,
        params_.joint_limit_stiffness,
        params_.joint_limit_damping,
        params_.joint_limit_max_torque);
  }

  Vector7d tau_d = tau_task + tau_nullspace + tau_limit + coriolis;
  tau_d = franky::saturateTorqueRate(tau_d, robot_state.tau_J_d, params_.max_delta_tau);

  std::array<double, 7> tau_d_array{};
  Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_d;

  auto output = franka::Torques(tau_d_array);
  if (finish) output = franka::MotionFinished(output);

  return output;
}

}  // namespace franky
