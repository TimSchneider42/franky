#include "franky/motion/joint_impedance_base.hpp"

#include <array>
#include <cmath>

#include "franky/model.hpp"
#include "franky/motion/torque_control_utils.hpp"

namespace franky {

JointImpedanceBase::JointImpedanceBase(
    const Vector7d &target, const Vector7d &target_velocity, const JointImpedanceParams &params,
    double gains_time_constant)
    : Motion<franka::Torques>(),
      params_(params),
      target_(target),
      target_velocity_(target_velocity),
      gains_handle_(JointImpedanceGains(params.stiffness, params.damping)),
      cartesian_gains_handle_(params.cartesian_gains.value_or(CartesianImpedanceGains{})),
      gains_time_constant_(gains_time_constant),
      current_stiffness_(params.stiffness),
      current_damping_(params.damping.value_or(defaultJointImpedanceDamping(params.stiffness))) {
  if (!std::isfinite(gains_time_constant_) || gains_time_constant_ <= 0.0) {
    throw std::invalid_argument("gains_time_constant must be finite and positive");
  }
  if (params.cartesian_gains.has_value()) {
    const auto &cartesian_gains = *params.cartesian_gains;
    CartesianShapingState shaping;
    shaping.stiffness = cartesian_gains.stiffness;
    shaping.critical_damping = defaultCartesianImpedanceDamping(cartesian_gains.stiffness);
    shaping.critical_damping_stiffness = cartesian_gains.stiffness;
    shaping.damping = cartesian_gains.damping.value_or(shaping.critical_damping);
    cartesian_shaping_ = shaping;
  }
  params_.validate();
}

const Matrix6d &JointImpedanceBase::criticalShapingDamping(CartesianShapingState &shaping) {
  // Recompute the critical-damping eigendecomposition only while the stiffness moves; cache otherwise.
  if (!shaping.critical_damping_stiffness.has_value() ||
      (shaping.critical_damping_stiffness->array() != shaping.stiffness.array()).any()) {
    shaping.critical_damping = defaultCartesianImpedanceDamping(shaping.stiffness);
    shaping.critical_damping_stiffness = shaping.stiffness;
  }
  return shaping.critical_damping;
}

franka::Torques JointImpedanceBase::computeCommand(
    const RobotState &robot_state, const JointReference &reference, double dt) {
  // Interpolate toward the target gains.
  const auto target_gains = gains_handle_.getUnsafe();
  const double alpha = 1.0 - std::exp(-dt / gains_time_constant_);
  current_stiffness_ += alpha * (target_gains.stiffness - current_stiffness_);
  // An unset damping target means "critically damp the current stiffness"
  const Vector7d target_damping =
      target_gains.damping.has_value() ? *target_gains.damping : defaultJointImpedanceDamping(current_stiffness_);
  current_damping_ += alpha * (target_damping - current_damping_);

  Vector7d torque_feedforward = params_.constant_torque_offset + reference.tau_ff;
  const Vector7d q_error = (reference.q - robot_state.q).cwiseMax(-params_.error_clip).cwiseMin(params_.error_clip);
  auto model = robot()->model();

  Vector7d tau_d;
  if (cartesian_shaping_.has_value()) {
    auto &shaping = *cartesian_shaping_;
    const auto target_gains = cartesian_gains_handle_.getUnsafe();
    shaping.stiffness = interpolateGain(shaping.stiffness, target_gains.stiffness, alpha);
    // An unset target means "critically damp the current stiffness"; interpolate toward it like any
    // other gain so unsetting damping is as smooth as setting it. The ternary keeps the
    // eigendecomposition off the explicit-damping path.
    const Matrix6d &target_damping =
        target_gains.damping.has_value() ? *target_gains.damping : criticalShapingDamping(shaping);
    shaping.damping += alpha * (target_damping - shaping.damping);

    const Jacobian jacobian = model->zeroJacobian(franka::Frame::kEndEffector, robot_state);
    const Eigen::Matrix<double, 7, 7> stiffness_eff =
        current_stiffness_.asDiagonal().toDenseMatrix() + jacobian.transpose() * shaping.stiffness * jacobian;
    const Eigen::Matrix<double, 7, 7> damping_eff =
        current_damping_.asDiagonal().toDenseMatrix() + jacobian.transpose() * shaping.damping * jacobian;
    tau_d = stiffness_eff * q_error + damping_eff * (reference.dq - robot_state.dq) + torque_feedforward;
  } else {
    tau_d = current_stiffness_.asDiagonal() * q_error +
            current_damping_.asDiagonal() * (reference.dq - robot_state.dq) + torque_feedforward;
  }

  tau_d += computeFrictionCompensation(robot_state.dq, params_.friction);

  if (params_.safety.lower_joint_limits.has_value() && params_.safety.upper_joint_limits.has_value()) {
    tau_d += franky::computeJointLimitTorque(
        robot_state.q,
        robot_state.dq,
        *params_.safety.lower_joint_limits,
        *params_.safety.upper_joint_limits,
        params_.safety.joint_limit_activation_distance,
        params_.safety.joint_limit_stiffness,
        params_.safety.joint_limit_damping,
        params_.safety.joint_limit_max_torque);
  }

  if (params_.compensate_coriolis) tau_d += model->coriolis(robot_state);
  tau_d = franky::saturateTorqueRate(tau_d, robot_state.tau_J_d, params_.safety.max_delta_tau);

  std::array<double, 7> tau_d_array{};
  Eigen::VectorXd::Map(tau_d_array.data(), 7) = tau_d;
  return franka::Torques(tau_d_array);
}

}  // namespace franky
