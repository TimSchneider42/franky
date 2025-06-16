#pragma once

#include <franka/model.h>

#include <franky/robot_state.hpp>

#include "franky/types.hpp"

namespace franky {

/**
 * @brief A wrapper around franka::Model that uses Eigen types.
 *
 * This class exposes the same functionality as franka::Model, but uses Eigen
 * types for inputs and outputs instead of std::array. All 2D arrays are
 * returned as Eigen matrices.
 */
class Model {
 public:
  /**
   * @param model The underlying franka::Model instance to wrap.
   */
  explicit Model(franka::Model model);

  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;

  Model(Model &&other) noexcept;
  Model &operator=(Model &&other) noexcept;

  /**
   * @brief Calculates the pose of a frame relative to the base frame.
   *
   * @param frame The frame whose pose should be returned.
   * @param state The current robot state.
   * @return The pose as an affine transformation matrix.
   */
  [[nodiscard]] Affine pose(franka::Frame frame, const RobotState &state) const;

  /**
   * @brief Calculates the pose of a frame relative to the base frame.
   *
   * @param frame The frame whose pose should be returned.
   * @param q Robot joint angles [rad].
   * @param F_T_EE Transformation from flange to end-effector frame.
   * @param EE_T_K Transformation from end-effector frame to stiffness frame.
   * @return The pose as an affine transformation matrix.
   */
  [[nodiscard]] Affine pose(franka::Frame frame, const Vector7d &q, const Affine &F_T_EE, const Affine &EE_T_K) const;

  /**
   * @brief Calculates the body Jacobian in base frame.
   *
   * @param frame The frame for which the Jacobian is computed.
   * @param state The current robot state.
   * @return The 6x7 body Jacobian matrix.
   */
  [[nodiscard]] Jacobian bodyJacobian(franka::Frame frame, const RobotState &state) const;

  /**
   * @brief Calculates the body Jacobian in base frame.
   *
   * @param frame The frame for which the Jacobian is computed.
   * @param q Robot joint angles [rad].
   * @param F_T_EE Transformation from flange to end-effector frame.
   * @param EE_T_K Transformation from end-effector frame to stiffness frame.
   * @return The 6x7 body Jacobian matrix.
   */
  [[nodiscard]] Jacobian bodyJacobian(
      franka::Frame frame, const Vector7d &q, const Affine &F_T_EE, const Affine &EE_T_K) const;

  /**
   * @brief Calculates the zero Jacobian in base frame.
   *
   * @param frame The frame for which the Jacobian is computed.
   * @param state The current robot state.
   * @return The 6x7 zero Jacobian matrix.
   */
  [[nodiscard]] Jacobian zeroJacobian(franka::Frame frame, const RobotState &state) const;

  /**
   * @brief Calculates the zero Jacobian in base frame.
   *
   * @param frame The frame for which the Jacobian is computed.
   * @param q Robot joint angles [rad].
   * @param F_T_EE Transformation from flange to end-effector frame.
   * @param EE_T_K Transformation from end-effector frame to stiffness frame.
   * @return The 6x7 zero Jacobian matrix.
   */
  [[nodiscard]] Jacobian zeroJacobian(
      franka::Frame frame, const Vector7d &q, const Affine &F_T_EE, const Affine &EE_T_K) const;

  /**
   * @brief Calculates the mass matrix.
   *
   * @param state The current robot state.
   * @return The 7x7 mass matrix.
   */
  [[nodiscard]] Eigen::Matrix<double, 7, 7> mass(const RobotState &state) const;

  /**
   * @brief Calculates the mass matrix.
   *
   * @param q Robot joint angles [rad].
   * @param I_total Combined load and robot inertia [kg·m²].
   * @param m_total Combined mass of robot and load [kg].
   * @param F_x_Ctotal Center of mass relative to flange frame [m].
   * @return The 7x7 mass matrix.
   */
  [[nodiscard]] Eigen::Matrix<double, 7, 7> mass(
      const Vector7d &q, const Eigen::Matrix3d &I_total, double m_total, const Eigen::Vector3d &F_x_Ctotal) const;

  /**
   * @brief Calculates the Coriolis force vector.
   *
   * @param state The current robot state.
   * @return The Coriolis vector [Nm].
   */
  [[nodiscard]] Vector7d coriolis(const RobotState &state) const;

  /**
   * @brief Calculates the Coriolis force vector.
   *
   * @param q Robot joint angles [rad].
   * @param dq Robot joint velocities [rad/s].
   * @param I_total Combined load and robot inertia [kg·m²].
   * @param m_total Combined mass of robot and load [kg].
   * @param F_x_Ctotal Center of mass relative to flange frame [m].
   * @return The Coriolis vector [Nm].
   */
  [[nodiscard]] Vector7d coriolis(
      const Vector7d &q, const Vector7d &dq, const Eigen::Matrix3d &I_total, double m_total,
      const Eigen::Vector3d &F_x_Ctotal) const;

  /**
   * @brief Calculates the gravity vector.
   *
   * @param state The current robot state.
   * @param gravity_earth Gravity vector in base frame [m/s²].
   * @return The gravity vector [Nm].
   */
  [[nodiscard]] Vector7d gravity(const RobotState &state, const Eigen::Vector3d &gravity_earth) const;

  /**
   * @brief Calculates the gravity vector using default gravity direction (0, 0,
   * -9.81).
   *
   * @param state The current robot state.
   * @return The gravity vector [Nm].
   */
  [[nodiscard]] Vector7d gravity(const RobotState &state) const;

  /**
   * @brief Calculates the gravity vector.
   *
   * @param q Robot joint angles [rad].
   * @param m_total Combined mass of robot and load [kg].
   * @param F_x_Ctotal Center of mass relative to flange frame [m].
   * @param gravity_earth Gravity vector in base frame [m/s²], default is (0, 0,
   * -9.81).
   * @return The gravity vector [Nm].
   */
  [[nodiscard]] Vector7d gravity(
      const Vector7d &q, double m_total, const Eigen::Vector3d &F_x_Ctotal,
      const Eigen::Vector3d &gravity_earth = {0., 0., -9.81}) const;

 private:
  franka::Model model_;
};

}  // namespace franky
