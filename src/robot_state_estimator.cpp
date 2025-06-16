#include "franky/robot_state_estimator.hpp"

#include <franky/model.hpp>
#include <franky/robot_state.hpp>

namespace franky {
RobotStateEstimator::RobotStateEstimator(
    double q_process_var, double dq_process_var, double ddq_process_var, double control_process_var, double q_obs_var,
    double dq_obs_var, double q_d_obs_var, double dq_d_obs_var, double ddq_d_obs_var, double control_adaptation_rate)
    : q_process_var_(q_process_var),
      dq_process_var_(dq_process_var),
      ddq_process_var_(ddq_process_var),
      control_process_var_(control_process_var),
      control_adaptation_rate_(control_adaptation_rate),
      prev_time_(std::nullopt) {
  // State format: q, dq, ddq, q_d - q, dq_d - dq, ddq_d - ddq
  // Observation format: q, dq, q_d, dq_d, ddq_d

  // clang-format off
  h_mat_ << 1, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0,
            1, 0, 0, 1, 0, 0,
            0, 1, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 1;
  // clang-format on

  Eigen::Vector<double, 5> r_diag;
  r_diag << q_obs_var, dq_obs_var, q_d_obs_var, dq_d_obs_var, ddq_d_obs_var;
  r_mat_ = Eigen::DiagonalMatrix<double, 5>(r_diag);
}

RobotState RobotStateEstimator::update(const franka::RobotState &franka_robot_state, const Model &model) {
  const auto q = toEigenD(franka_robot_state.q);
  const auto dq = toEigenD(franka_robot_state.dq);
  const auto q_d = toEigenD(franka_robot_state.q_d);
  const auto dq_d = toEigenD(franka_robot_state.dq_d);
  const auto ddq_d = toEigenD(franka_robot_state.ddq_d);

  Eigen::Matrix<double, 7, 5> observation;
  observation << q, dq, q_d, dq_d, ddq_d;

  if (prev_time_.has_value()) {
    const auto dt = (franka_robot_state.time - prev_time_.value()).toSec();

    // clang-format off
    const auto lambda = control_adaptation_rate_;
    Eigen::Matrix<double, 6, 6> f_mat;
    f_mat << 1, dt, 0.5 * std::pow(dt, 2), lambda,     0,          0,
             0, 1,  dt,                      0,          lambda,     0,
             0, 0,  1,                       0,          0,          lambda,
             0, 0,  0,                       1 - lambda, 0,          0,
             0, 0,  0,                       0,          1 - lambda, 0,
             0, 0,  0,                       0,          0,          1 - lambda;


    Eigen::Vector3d g_mat_acc;
    g_mat_acc << 0.5 * std::pow(dt, 2), dt, 1;

    Eigen::Vector3d g_mat_vel;
    g_mat_vel << dt, 1, 0;

    Eigen::Vector3d g_mat_pos;
    g_mat_pos << 1, 0, 0;

    const auto q_mat_s = q_process_var_ * (g_mat_pos * g_mat_pos.transpose()) +
                         dq_process_var_ * (g_mat_vel * g_mat_vel.transpose()) +
                         ddq_process_var_ * (g_mat_acc * g_mat_acc.transpose());

    Eigen::Matrix<double, 6, 6> q_mat;
    q_mat << q_mat_s,                 Eigen::Matrix3d::Zero(),
             Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Identity() * control_process_var_;
    // clang-format on

    // Predict
    joint_state_mean_ = f_mat * joint_state_mean_;

    for (int i = 0; i < 7; i++) {
      // Predict
      joint_state_covar_[i] = f_mat * joint_state_covar_[i] * f_mat.transpose() + q_mat;

      // Observe
      const auto s_mat = h_mat_ * joint_state_covar_[i] * h_mat_.transpose() + r_mat_;
      const auto k_mat = joint_state_covar_[i] * h_mat_.transpose() * s_mat.inverse();

      const auto observation_error = observation.transpose().col(i) - h_mat_ * joint_state_mean_.col(i);
      joint_state_mean_.col(i) = joint_state_mean_.col(i) + k_mat * observation_error;
      joint_state_covar_[i] = (Eigen::Matrix<double, 6, 6>::Identity() - k_mat * h_mat_) * joint_state_covar_[i];
    }
  } else {
    // Use MAP estimate
    for (int i = 0; i < 7; i++) {
      const auto r_mat_inv = r_mat_.inverse();
      joint_state_covar_[i] =
          (h_mat_.transpose() * r_mat_inv * h_mat_ + Eigen::Matrix<double, 6, 6>::Identity() * 1e-4).inverse();
      joint_state_mean_.col(i) =
          joint_state_covar_[i] * h_mat_.transpose() * r_mat_inv * observation.transpose().col(i);
    }
  }

  const auto ee_jacobian = model.bodyJacobian(
      franka::Frame::kEndEffector,
      toEigenD(franka_robot_state.q),
      stdToAffine(franka_robot_state.F_T_EE),
      stdToAffine(franka_robot_state.EE_T_K));
  prev_time_ = franka_robot_state.time;
  return RobotState::from_franka(
      franka_robot_state, ee_jacobian, joint_state_mean_.row(0), joint_state_mean_.row(1), joint_state_mean_.row(2));
}

}  // namespace franky
