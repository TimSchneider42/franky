#pragma once

#include <algorithm>
#include <cmath>
#include <optional>

#include "franky/types.hpp"

namespace franky {

struct TorqueSafetyParams {
  /** Maximum allowed torque step per cycle in [Nm]. */
  double max_delta_tau{1.0};

  /** Lower soft joint limits in [rad]. Joint-limit repulsion is active when both limits are set. */
  std::optional<Vector7d> lower_joint_limits{};

  /** Upper soft joint limits in [rad]. Joint-limit repulsion is active when both limits are set. */
  std::optional<Vector7d> upper_joint_limits{};

  /** Activation distance from a limit in [rad]. */
  double joint_limit_activation_distance{0.1};

  /** Base repulsion gain in [Nm]. */
  double joint_limit_stiffness{4.0};

  /** Additional damping when moving into a limit in [Nms/rad]. */
  double joint_limit_damping{1.0};

  /** Absolute torque clamp for the repulsion term in [Nm]. */
  double joint_limit_max_torque{5.0};
};

struct FrictionCompensationParams {
  /** Coulomb friction compensation gains in [Nm]. */
  Vector7d coulomb{Vector7d::Zero()};

  /** Viscous friction compensation gains in [Nms/rad]. */
  Vector7d viscous{Vector7d::Zero()};

  /** Absolute per-joint clamp for friction compensation in [Nm]. */
  Vector7d max_torque{Vector7d::Ones()};

  /** Velocity scale for the smooth Coulomb sign transition in [rad/s]. */
  double velocity_epsilon{0.03};
};

inline Vector7d saturateTorqueRate(
    const Vector7d &tau_d_calculated, const Vector7d &tau_reference, double max_delta_tau) {
  Vector7d tau_d_saturated;
  for (size_t i = 0; i < 7; i++) {
    const double difference = tau_d_calculated[i] - tau_reference[i];
    tau_d_saturated[i] = tau_reference[i] + std::max(std::min(difference, max_delta_tau), -max_delta_tau);
  }
  return tau_d_saturated;
}

inline Vector7d computeJointLimitTorque(
    const Vector7d &q, const Vector7d &dq, const Vector7d &lower_joint_limits, const Vector7d &upper_joint_limits,
    double joint_limit_activation_distance, double joint_limit_stiffness, double joint_limit_damping,
    double joint_limit_max_torque) {
  Vector7d tau_limit = Vector7d::Zero();
  const double activation_distance = std::max(joint_limit_activation_distance, 1e-6);

  for (size_t i = 0; i < 7; i++) {
    const double lower_soft_limit = lower_joint_limits[i] + activation_distance;
    const double upper_soft_limit = upper_joint_limits[i] - activation_distance;

    if (q[i] < lower_soft_limit) {
      const double penetration = (lower_soft_limit - q[i]) / activation_distance;
      double tau = joint_limit_stiffness * (std::exp(penetration) - 1.0);
      if (dq[i] < 0.0) tau += joint_limit_damping * (-dq[i]);
      tau_limit[i] = std::min(tau, joint_limit_max_torque);
    } else if (q[i] > upper_soft_limit) {
      const double penetration = (q[i] - upper_soft_limit) / activation_distance;
      double tau = -joint_limit_stiffness * (std::exp(penetration) - 1.0);
      if (dq[i] > 0.0) tau -= joint_limit_damping * dq[i];
      tau_limit[i] = std::max(tau, -joint_limit_max_torque);
    }
  }

  return tau_limit;
}

inline Vector7d computeFrictionCompensation(const Vector7d &dq, const FrictionCompensationParams &params) {
  Vector7d tau = Vector7d::Zero();
  const double epsilon = std::max(params.velocity_epsilon, 1e-9);
  for (int i = 0; i < 7; ++i) {
    const double limit = std::max(params.max_torque[i], 0.0);
    const double uncompensated = params.coulomb[i] * std::tanh(dq[i] / epsilon) + params.viscous[i] * dq[i];
    tau[i] = std::clamp(uncompensated, -limit, limit);
  }
  return tau;
}

}  // namespace franky
