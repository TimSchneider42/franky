#pragma once

#include <franka/control_types.h>

#include <Eigen/Core>
#include <optional>

#include "franky/elbow_state.hpp"
#include "franky/twist.hpp"
#include "franky/types.hpp"

namespace franky {

/**
 * @brief Cartesian velocity of a robot.
 *
 * This class encapsulates the cartesian velocity of a robot, which comproses
 * the end effector twist and the elbow velocity.
 */
class RobotVelocity {
 public:
  RobotVelocity();

  RobotVelocity(const RobotVelocity &robot_velocity);

  // Suppress implicit conversion warning
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-conversion"
  /**
   * @param end_effector_twist The twist of the end effector.
   * @param elbow_velocity The velocity of the elbow (optional).
   */
  RobotVelocity(const Twist &end_effector_twist, std::optional<double> elbow_velocity = std::nullopt);
#pragma clang diagnostic pop

  /**
   * @param vector_repr  The vector representation of the velocity.
   * @param ignore_elbow Whether to ignore the elbow velocity. Default is false.
   */
  explicit RobotVelocity(const Vector7d &vector_repr, bool ignore_elbow = false);

  /**
   * @param vector_repr The vector representation of the velocity.
   * @param elbow_velocity The velocity of the elbow (optional).
   */
  explicit RobotVelocity(const Vector6d &vector_repr, std::optional<double> elbow_velocity = std::nullopt);

  /**
   * @param franka_velocity The franka velocity.
   */
  explicit RobotVelocity(franka::CartesianVelocities franka_velocity);

  /**
   * @brief Get the vector representation of the velocity. It consists of the
   * linear and angular velocity of the end effector and the joint velocity of
   * the elbow.
   *
   * @return The vector representation of the velocity.
   */
  [[nodiscard]] Vector7d vector_repr() const;

  /**
   * @brief Get the franka velocity.
   *
   * @param elbow_state                  The elbow state to use. Note, that
   * franka::CartesianVelocities contains the elbow state and not the elbow
   * velocity, contrary to RobotVelocity.
   * @param default_elbow_flip_direction The default flip direction of the elbow
   * if it is not set.
   * @return The franka velocity.
   */
  [[nodiscard]] franka::CartesianVelocities as_franka_velocity(
      const std::optional<ElbowState> &elbow_state = std::nullopt,
      FlipDirection default_elbow_flip_direction = FlipDirection::kNegative) const;

  /**
   * @brief Transform the frame of the velocity by applying the given affine
   * transform.
   *
   * @param affine The affine to apply.
   * @return The velocity after the transformation.
   */
  [[nodiscard]] inline RobotVelocity transform(const Affine &affine) const { return transform(affine.rotation()); }

  /**
   * @brief Transform the frame of the velocity by applying the given rotation.
   *
   * @param rotation The rotation to apply.
   * @return The velocity after the transformation.
   */
  template <typename RotationMatrixType>
  [[nodiscard]] inline RobotVelocity transform(const RotationMatrixType &rotation) const {
    return {rotation * end_effector_twist_, elbow_velocity_};
  }

  /**
   * @brief Change the end-effector frame by adding the given offset to the
   * current end-effector frame. Note that the offset must be given in world
   * coordinates.
   *
   * @param offset_world_frame The offset to add to the current end-effector
   * frame.
   * @return The velocity of the new end-effector frame.
   */
  [[nodiscard]] inline RobotVelocity changeEndEffectorFrame(const Eigen::Vector3d &offset_world_frame) const {
    return {end_effector_twist_.propagateThroughLink(offset_world_frame), elbow_velocity_};
  }

  /**
   * @brief Get the velocity with a new elbow velocity.
   *
   * @param elbow_velocity The new elbow velocity.
   * @return The velocity with the new elbow velocity.
   */
  [[nodiscard]] inline RobotVelocity withElbowVelocity(const std::optional<double> elbow_velocity) const {
    return {end_effector_twist_, elbow_velocity};
  }

  /**
   * @brief Get the end effector twist.
   *
   * @return The end effector twist.
   */
  [[nodiscard]] inline Twist end_effector_twist() const { return end_effector_twist_; }

  /**
   * @brief Get the elbow velocity.
   *
   * @return The elbow velocity.
   */
  [[nodiscard]] inline std::optional<double> elbow_velocity() const { return elbow_velocity_; }

  friend std::ostream &operator<<(std::ostream &os, const RobotVelocity &robot_velocity);

 private:
  Twist end_effector_twist_;
  std::optional<double> elbow_velocity_ = std::nullopt;
};

inline RobotVelocity operator*(const Affine &affine, const RobotVelocity &robot_velocity) {
  return robot_velocity.transform(affine);
}

template <typename RotationMatrixType>
inline RobotVelocity operator*(const RotationMatrixType &rotation, const RobotVelocity &robot_velocity) {
  return robot_velocity.transform(rotation);
}

}  // namespace franky
