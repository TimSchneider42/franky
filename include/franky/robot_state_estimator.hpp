#pragma once

#include "franky/model.hpp"
#include "franky/robot_state.hpp"

namespace franky {

class RobotStateEstimator {
 public:
  RobotStateEstimator(
      double q_process_var, double dq_process_var, double ddq_process_var, double control_process_var, double q_obs_var,
      double dq_obs_var, double q_d_obs_var, double dq_d_obs_var, double ddq_d_obs_var, double control_adaptation_rate);

  RobotStateEstimator(const RobotStateEstimator &) = default;

  RobotState update(const franka::RobotState &franka_robot_state, const Model &model);

  RobotState operator()(const franka::RobotState &franka_robot_state, const Model &model) {
    return update(franka_robot_state, model);
  }

 private:
  double q_process_var_;
  double dq_process_var_;
  double ddq_process_var_;
  double control_process_var_;
  double control_adaptation_rate_;

  Eigen::Matrix<double, 5, 6> h_mat_;
  Eigen::Matrix<double, 5, 5> r_mat_;

  std::optional<franka::Duration> prev_time_{};
  Eigen::Matrix<double, 6, 7> joint_state_mean_;
  std::array<Eigen::Matrix<double, 6, 6>, 7> joint_state_covar_;
};

}  // namespace franky
