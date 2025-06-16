// Adapted from
// https://github.com/frankaemika/libfranka/blob/main/include/franka/robot_state.h
#pragma once

#include <franka/robot_state.h>

#include "franky/elbow_state.hpp"
#include "franky/twist.hpp"
#include "franky/twist_acceleration.hpp"
#include "franky/types.hpp"
#include "franky/util.hpp"

namespace franky {

/**
 * @brief Full state of the robot
 *
 * This class contains all fields of franka::RobotState and some additional
 * fields. Each additional field ends in
 * "_est". Unlike franka::RobotState, all fields are converted to appropriate
 * Eigen/franky types.
 */
struct RobotState {
  static RobotState from_franka(
      const franka::RobotState &robot_state, std::optional<Vector7d> q_est = std::nullopt,
      std::optional<Vector7d> dq_est = std::nullopt, std::optional<Vector7d> ddq_est = std::nullopt,
      std::optional<Twist> O_dP_EE_est = std::nullopt, std::optional<TwistAcceleration> O_ddP_EE_est = std::nullopt,
      std::optional<double> delbow_est = std::nullopt, std::optional<double> ddelbow_est = std::nullopt);

  static RobotState from_franka(
      const franka::RobotState &robot_state, const Jacobian &ee_jacobian, const Vector7d &q_est, const Vector7d &dq_est,
      const Vector7d &ddq_est);

  /**
   * \f$^{O}T_{EE}\f$
   * Measured end effector pose in @ref o-frame "base frame".
   */
  Affine O_T_EE{};  // NOLINT(readability-identifier-naming)

  /**
   * \f${^OT_{EE}}_{d}\f$
   * Last desired end effector pose of motion generation in @ref o-frame "base
   * frame".
   */
  Affine O_T_EE_d{};  // NOLINT(readability-identifier-naming)

  /**
   * \f$^{F}T_{EE}\f$
   * End effector frame pose in flange frame.
   *
   * @see F_T_NE
   * @see NE_T_EE
   * @see Robot for an explanation of the F, NE and EE frames.
   */
  Affine F_T_EE{};  // NOLINT(readability-identifier-naming)

#ifdef FRANKA_0_8
  /**
   * \f$^{F}T_{NE}\f$
   * Nominal end effector frame pose in flange frame.
   *
   * @see F_T_EE
   * @see NE_T_EE
   * @see Robot for an explanation of the F, NE and EE frames.
   */
  Affine F_T_NE{};  // NOLINT(readability-identifier-naming)

  /**
   * \f$^{NE}T_{EE}\f$
   * End effector frame pose in nominal end effector frame.
   *
   * @see Robot::setEE to change this frame.
   * @see F_T_EE
   * @see F_T_NE
   * @see Robot for an explanation of the F, NE and EE frames.
   */
  Affine NE_T_EE{};  // NOLINT(readability-identifier-naming)
#endif

  /**
   * \f$^{EE}T_{K}\f$
   * Stiffness frame pose in end effector frame.
   *
   * See also @ref k-frame "K frame".
   */
  Affine EE_T_K{};  // NOLINT(readability-identifier-naming)

  /**
   * \f$m_{EE}\f$
   * Configured mass of the end effector.
   */
  double m_ee{};

  /**
   * \f$I_{EE}\f$
   * Configured rotational inertia matrix of the end effector load with respect
   * to center of mass.
   */
  IntertiaMatrix I_ee{};  // NOLINT(readability-identifier-naming)

  /**
   * \f$^{F}x_{C_{EE}}\f$
   * Configured center of mass of the end effector load with respect to flange
   * frame.
   */
  Eigen::Vector3d F_x_Cee{};  // NOLINT(readability-identifier-naming)

  /**
   * \f$m_{load}\f$
   * Configured mass of the external load.
   */
  double m_load{};

  /**
   * \f$I_{load}\f$
   * Configured rotational inertia matrix of the external load with respect to
   * center of mass.
   */
  IntertiaMatrix I_load{};  // NOLINT(readability-identifier-naming)

  /**
   * \f$^{F}x_{C_{load}}\f$
   * Configured center of mass of the external load with respect to flange
   * frame.
   */
  Eigen::Vector3d F_x_Cload{};  // NOLINT(readability-identifier-naming)

  /**
   * \f$m_{total}\f$
   * Sum of the mass of the end effector and the external load.
   */
  double m_total{};

  /**
   * \f$I_{total}\f$
   * Combined rotational inertia matrix of the end effector load and the
   * external load with respect to the center of mass.
   */
  IntertiaMatrix I_total{};  // NOLINT(readability-identifier-naming)

  /**
   * \f$^{F}x_{C_{total}}\f$
   * Combined center of mass of the end effector load and the external load with
   * respect to flange frame.
   */
  Eigen::Vector3d F_x_Ctotal{};  // NOLINT(readability-identifier-naming)

  /**
   * Elbow configuration.
   */
  ElbowState elbow{};

  /**
   * Desired elbow configuration.
   */
  ElbowState elbow_d{};

  /**
   * Commanded elbow configuration.
   */
  ElbowState elbow_c{};

  /**
   * Commanded velocity of the 3rd joint in \f$\frac{rad}{s}\f$
   */
  double delbow_c{};

  /**
   * Commanded elbow acceleration of the 3rd joint in \f$\frac{rad}{s^2}\f$
   */
  double ddelbow_c{};

  /**
   * \f$\tau_{J}\f$
   * Measured link-side joint torque sensor signals. Unit: \f$[Nm]\f$
   */
  Vector7d tau_J{};  // NOLINT(readability-identifier-naming)

  /**
   * \f${\tau_J}_d\f$
   * Desired link-side joint torque sensor signals without gravity. Unit:
   * \f$[Nm]\f$
   */
  Vector7d tau_J_d{};  // NOLINT(readability-identifier-naming)

  /**
   * \f$\dot{\tau_{J}}\f$
   * Derivative of measured link-side joint torque sensor signals. Unit:
   * \f$[\frac{Nm}{s}]\f$
   */
  Vector7d dtau_J{};  // NOLINT(readability-identifier-naming)

  /**
   * \f$q\f$
   * Measured joint position. Unit: \f$[rad]\f$
   */
  Vector7d q{};

  /**
   * \f$q_d\f$
   * Desired joint position. Unit: \f$[rad]\f$
   */
  Vector7d q_d{};

  /**
   * \f$\dot{q}\f$
   * Measured joint velocity. Unit: \f$[\frac{rad}{s}]\f$
   */
  Vector7d dq{};

  /**
   * \f$\dot{q}_d\f$
   * Desired joint velocity. Unit: \f$[\frac{rad}{s}]\f$
   */
  Vector7d dq_d{};

  /**
   * \f$\ddot{q}_d\f$
   * Desired joint acceleration. Unit: \f$[\frac{rad}{s^2}]\f$
   */
  Vector7d ddq_d{};

  /**
   * Indicates which contact level is activated in which joint. After contact
   * disappears, value turns to zero.
   *
   * @see Robot::setCollisionBehavior for setting sensitivity values.
   */
  Vector7d joint_contact{};

  /**
   * Indicates which contact level is activated in which Cartesian dimension
   * \f$(x,y,z,R,P,Y)\f$. After contact disappears, the value turns to zero.
   *
   * @see Robot::setCollisionBehavior for setting sensitivity values.
   */
  Vector6d cartesian_contact{};

  /**
   * Indicates which contact level is activated in which joint. After contact
   * disappears, the value stays the same until a reset command is sent.
   *
   * @see Robot::setCollisionBehavior for setting sensitivity values.
   * @see Robot::automaticErrorRecovery for performing a reset after a
   * collision.
   */
  Vector7d joint_collision{};

  /**
   * Indicates which contact level is activated in which Cartesian dimension
   * \f$(x,y,z,R,P,Y)\f$. After contact disappears, the value stays the same
   * until a reset command is sent.
   *
   * @see Robot::setCollisionBehavior for setting sensitivity values.
   * @see Robot::automaticErrorRecovery for performing a reset after a
   * collision.
   */
  Vector6d cartesian_collision{};

  /**
   * \f$\hat{\tau}_{\text{ext}}\f$
   * Low-pass filtered torques generated by external forces on the joints. It
   * does not include configured end-effector and load nor the mass and dynamics
   * of the robot. tau_ext_hat_filtered is the error between tau_J and the
   * expected torques given by the robot model. Unit: \f$[Nm]\f$.
   */
  Vector7d tau_ext_hat_filtered{};

  /**
   * \f$^OF_{K,\text{ext}}\f$
   * Estimated external wrench (force, torque) acting on stiffness frame,
   * expressed relative to the @ref o-frame "base frame". Forces applied by the
   * robot to the environment are positive, while forces applied by the
   * environment on the robot are negative. Becomes
   * \f$[0,0,0,0,0,0]\f$ when near or in a singularity. See also @ref k-frame
   * "Stiffness frame K". Unit: \f$[N,N,N,Nm,Nm,Nm]\f$.
   */
  Vector6d O_F_ext_hat_K{};  // NOLINT(readability-identifier-naming)

  /**
   * \f$^{K}F_{K,\text{ext}}\f$
   * Estimated external wrench (force, torque) acting on stiffness frame,
   * expressed relative to the stiffness frame. Forces applied by the robot to
   * the environment are positive, while forces applied by the environment on
   * the robot are negative. Becomes
   * \f$[0,0,0,0,0,0]\f$ when near or in a singularity. See also @ref k-frame
   * "Stiffness frame K". Unit: \f$[N,N,N,Nm,Nm,Nm]\f$.
   */
  Vector6d K_F_ext_hat_K{};  // NOLINT(readability-identifier-naming)

  /**
   * \f${^OdP_{EE}}_{d}\f$
   * Desired end effector twist in @ref o-frame "base frame".
   * Unit:
   * \f$[\frac{m}{s},\frac{m}{s},\frac{m}{s},\frac{rad}{s},\frac{rad}{s},\frac{rad}{s}]\f$.
   */
  Twist O_dP_EE_d{};  // NOLINT(readability-identifier-naming)

#ifdef FRANKA_0_9
  /**
   * \f${^OddP}_O\f$
   * Linear component of the acceleration of the robot's base, expressed in
   * frame parallel to the
   * @ref o-frame "base frame", i.e. the base's translational acceleration. If
   * the base is resting this shows the direction of the gravity vector. It is
   * hardcoded for now to `{0, 0, -9.81}`.
   */
  Eigen::Vector3d O_ddP_O{};  // NOLINT(readability-identifier-naming)
#endif

  /**
   * \f${^OT_{EE}}_{c}\f$
   * Last commanded end effector pose of motion generation in @ref o-frame "base
   * frame".
   */
  Affine O_T_EE_c{};  // NOLINT(readability-identifier-naming)

  /**
   * \f${^OdP_{EE}}_{c}\f$
   * Last commanded end effector twist in @ref o-frame "base frame".
   */
  Twist O_dP_EE_c{};  // NOLINT(readability-identifier-naming)

  /**
   * \f${^OddP_{EE}}_{c}\f$
   * Last commanded end effector acceleration in @ref o-frame "base frame".
   * Unit:
   * \f$[\frac{m}{s^2},\frac{m}{s^2},\frac{m}{s^2},\frac{rad}{s^2},\frac{rad}{s^2},\frac{rad}{s^2}]\f$.
   */
  TwistAcceleration O_ddP_EE_c{};  // NOLINT(readability-identifier-naming)

  /**
   * \f$\theta\f$
   * Motor position. Unit: \f$[rad]\f$
   */
  Vector7d theta{};

  /**
   * \f$\dot{\theta}\f$
   * Motor velocity. Unit: \f$[\frac{rad}{s}]\f$
   */
  Vector7d dtheta{};

  /**
   * Current error state.
   */
  franka::Errors current_errors{};

  /**
   * Contains the errors that aborted the previous motion.
   */
  franka::Errors last_motion_errors{};

  /**
   * Percentage of the last 100 control commands that were successfully received
   * by the robot.
   *
   * Shows a value of zero if no control or motion generator loop is currently
   * running.
   *
   * Range: \f$[0, 1]\f$.
   */
  double control_command_success_rate{};

  /**
   * Current robot mode.
   */
  franka::RobotMode robot_mode = franka::RobotMode::kUserStopped;

  /**
   * Strictly monotonically increasing timestamp since robot start.
   *
   * Inside of control loops @ref callback-docs "time_step" parameter of
   * Robot::control can be used instead.
   */
  franka::Duration time{};

  /**
   * Estimated joint position computed by franky. This value does not come from
   * the franka firmware. Unit: \f$[rad]\f$.
   */
  std::optional<Vector7d> q_est{};

  /**
   * Estimated joint velocity computed by franky. This value does not come from
   * the franka firmware. Unit: \f$[\frac{rad}{s}]\f$.
   */
  std::optional<Vector7d> dq_est{};

  /**
   * Estimated joint acceleration computed by franky. This value does not come
   * from the franka firmware. Unit: \f$[\frac{rad}{s^2}]\f$.
   */
  std::optional<Vector7d> ddq_est{};

  /**
   * Estimated end-effector twist (linear and angular velocity) expressed in
   * @ref o-frame "base frame". Computed by franky; not provided by franka
   * firmware. Unit:
   * \f$[\frac{m}{s},\frac{m}{s},\frac{m}{s},\frac{rad}{s},\frac{rad}{s},\frac{rad}{s}]\f$.
   */
  std::optional<Twist> O_dP_EE_est{};

  /**
   * Estimated end-effector spatial acceleration (linear and angular
   * acceleration) expressed in
   * @ref o-frame "base frame". Computed by franky; not provided by franka
   * firmware. Unit:
   * \f$[\frac{m}{s^2},\frac{m}{s^2},\frac{m}{s^2},\frac{rad}{s^2},\frac{rad}{s^2},\frac{rad}{s^2}]\f$.
   */
  std::optional<TwistAcceleration> O_ddP_EE_est{};

  /**
   * Estimated elbow velocity (velocity of the third joint), computed by franky.
   * Not provided by franka firmware.
   * Unit: \f$[\frac{rad}{s}]\f$.
   */
  std::optional<double> delbow_est{};

  /**
   * Estimated elbow acceleration (acceleration of the third joint), computed by
   * franky. Not provided by franka firmware. Unit: \f$[\frac{rad}{s^2}]\f$.
   */
  std::optional<double> ddelbow_est{};
};

}  // namespace franky
