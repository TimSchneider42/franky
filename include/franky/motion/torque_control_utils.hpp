#pragma once

#include <algorithm>
#include <cmath>

#include "franky/types.hpp"

namespace franky {

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
    const Vector7d &q,
    const Vector7d &dq,
    const Vector7d &lower_joint_limits,
    const Vector7d &upper_joint_limits,
    double joint_limit_activation_distance,
    double joint_limit_stiffness,
    double joint_limit_damping,
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

}  // namespace franky
