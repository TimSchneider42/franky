#include "franky/robot_state.hpp"

#include "franky/util.hpp"

namespace franky {

RobotState RobotState::from_franka(
    const franka::RobotState &franka_robot_state, std::optional<Vector7d> q_est, std::optional<Vector7d> dq_est,
    std::optional<Vector7d> ddq_est, std::optional<Twist> O_dP_EE_est, std::optional<TwistAcceleration> O_ddP_EE_est,
    std::optional<double> delbow_est, std::optional<double> ddelbow_est) {
  return RobotState{
      .O_T_EE = stdToAffine(franka_robot_state.O_T_EE),
      .O_T_EE_d = stdToAffine(franka_robot_state.O_T_EE_d),
      .F_T_EE = stdToAffine(franka_robot_state.F_T_EE),
#ifdef FRANKA_0_8
      .F_T_NE = stdToAffine(franka_robot_state.F_T_NE),
      .NE_T_EE = stdToAffine(franka_robot_state.NE_T_EE),
#endif
      .EE_T_K = stdToAffine(franka_robot_state.EE_T_K),
      .m_ee = franka_robot_state.m_ee,
      .I_ee = toEigenMatD<3, 3>(franka_robot_state.I_ee),
      .F_x_Cee = toEigen(franka_robot_state.F_x_Cee),
      .m_load = franka_robot_state.m_load,
      .I_load = toEigenMatD<3, 3>(franka_robot_state.I_load),
      .F_x_Cload = toEigen(franka_robot_state.F_x_Cload),
      .m_total = franka_robot_state.m_total,
      .I_total = toEigenMatD<3, 3>(franka_robot_state.I_total),
      .F_x_Ctotal = toEigen(franka_robot_state.F_x_Ctotal),
      .elbow = ElbowState(franka_robot_state.elbow),
      .elbow_d = ElbowState(franka_robot_state.elbow_d),
      .elbow_c = ElbowState(franka_robot_state.elbow_c),
      .delbow_c = franka_robot_state.delbow_c[0],
      .ddelbow_c = franka_robot_state.ddelbow_c[0],
      .tau_J = toEigen(franka_robot_state.tau_J),
      .tau_J_d = toEigen(franka_robot_state.tau_J_d),
      .dtau_J = toEigen(franka_robot_state.dtau_J),
      .q = toEigen(franka_robot_state.q),
      .q_d = toEigen(franka_robot_state.q_d),
      .dq = toEigen(franka_robot_state.dq),
      .dq_d = toEigen(franka_robot_state.dq_d),
      .ddq_d = toEigen(franka_robot_state.ddq_d),
      .joint_contact = toEigen(franka_robot_state.joint_contact),
      .cartesian_contact = toEigen(franka_robot_state.cartesian_contact),
      .joint_collision = toEigen(franka_robot_state.joint_collision),
      .cartesian_collision = toEigen(franka_robot_state.cartesian_collision),
      .tau_ext_hat_filtered = toEigen(franka_robot_state.tau_ext_hat_filtered),
      .O_F_ext_hat_K = toEigen(franka_robot_state.O_F_ext_hat_K),
      .K_F_ext_hat_K = toEigen(franka_robot_state.K_F_ext_hat_K),
      .O_dP_EE_d = Twist::fromVectorRepr(toEigen(franka_robot_state.O_dP_EE_d)),
#ifdef FRANKA_0_9
      .O_ddP_O = toEigen(franka_robot_state.O_ddP_O),
#endif
      .O_T_EE_c = stdToAffine(franka_robot_state.O_T_EE_c),
      .O_dP_EE_c = Twist::fromVectorRepr(toEigen(franka_robot_state.O_dP_EE_c)),
      .O_ddP_EE_c = TwistAcceleration::fromVectorRepr(toEigen(franka_robot_state.O_ddP_EE_c)),
      .theta = toEigen(franka_robot_state.theta),
      .dtheta = toEigen(franka_robot_state.dtheta),
      .current_errors = franka_robot_state.current_errors,
      .last_motion_errors = franka_robot_state.last_motion_errors,
      .control_command_success_rate = franka_robot_state.control_command_success_rate,
      .robot_mode = franka_robot_state.robot_mode,
      .time = franka_robot_state.time,
      .q_est = std::move(q_est),
      .dq_est = std::move(dq_est),
      .ddq_est = std::move(ddq_est),
      .O_dP_EE_est = std::move(O_dP_EE_est),
      .O_ddP_EE_est = std::move(O_ddP_EE_est),
      .delbow_est = delbow_est,
      .ddelbow_est = ddelbow_est,
  };
}

RobotState RobotState::from_franka(
    const franka::RobotState &franka_robot_state, const Jacobian &ee_jacobian, const Vector7d &q_est,
    const Vector7d &dq_est, const Vector7d &ddq_est) {
  auto O_dp_EE_est_vec = ee_jacobian * dq_est;
  auto O_dp_EE_est = Twist::fromVectorRepr(O_dp_EE_est_vec);
  auto O_ddp_EE_est_vec = ee_jacobian * ddq_est;
  auto O_ddp_EE_est = TwistAcceleration::fromVectorRepr(O_ddp_EE_est_vec);
  return from_franka(franka_robot_state, q_est, dq_est, ddq_est, O_dp_EE_est, O_ddp_EE_est, dq_est[2], ddq_est[2]);
}

}  // namespace franky
