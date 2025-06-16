#pragma once

#include <franka/control_types.h>

#include <Eigen/Core>
#include <optional>

#include "franky/elbow_state.hpp"
#include "franky/types.hpp"

namespace franky {

/**
 * @brief Cartesian pose of a robot.
 *
 * This class encapsulates the cartesian pose of a robot, which comprises the
 * end effector pose and the elbow position.
 */
class RobotPose {
 public:
  RobotPose();

  RobotPose(const RobotPose &robot_pose);

  // Suppress implicit conversion warning
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-conversion"
  /**
   * @param end_effector_pose The pose of the end effector.
   * @param elbow_state       The state of the elbow. Optional.
   */
  RobotPose(Affine end_effector_pose, std::optional<ElbowState> elbow_state = std::nullopt);
#pragma clang diagnostic pop

  /**
   * @param vector_repr    The vector representation of the pose.
   * @param ignore_elbow   Whether to ignore the elbow state. Default is false.
   * @param flip_direction The flip direction to use if the elbow is not
   * ignored. Default is negative.
   */
  explicit RobotPose(
      const Vector7d &vector_repr, bool ignore_elbow = false, FlipDirection flip_direction = FlipDirection::kNegative);

  /**
   * @param vector_repr The vector representation of the pose.
   * @param elbow_state The state of the elbow. Optional.
   */
  explicit RobotPose(const Vector6d &vector_repr, std::optional<ElbowState> elbow_state = std::nullopt);

  /**
   * @param franka_pose The franka pose.
   */
  explicit RobotPose(const franka::CartesianPose &franka_pose);

  /**
   * @brief Get the vector representation of the pose, which consists of the end
   * effector position and orientation (as rotation vector) and the elbow
   * position. Does not contain the flip component of the elbow state.
   *
   * @return The vector representation of the pose.
   */
  [[nodiscard]] Vector7d vector_repr() const;

  /**
   * @brief Convert this pose to a franka pose.
   *
   * @param default_elbow_flip_direction The default flip direction to use if
   * the elbow flip direction is not set.
   * @return The franka pose.
   */
  [[nodiscard]] franka::CartesianPose as_franka_pose(
      FlipDirection default_elbow_flip_direction = FlipDirection::kNegative) const;

  /**
   * @brief Transform this pose with a given affine transformation from the left
   * side.
   *
   * @param transform The transform to apply.
   * @return The transformed robot pose.
   */
  [[nodiscard]] RobotPose leftTransform(const Affine &transform) const {
    return {transform * end_effector_pose_, elbow_state_};
  }

  /**
   * @brief Transform this pose with a given affine transformation from the
   * right side.
   *
   * @param transform The transform to apply.
   * @return The transformed robot pose.
   */
  [[nodiscard]] RobotPose rightTransform(const Affine &transform) const {
    return {end_effector_pose_ * transform, elbow_state_};
  }

  /**
   * @brief Change the frame of the end effector by applying a transformation
   * from the right side. This is equivalent to calling
   * rightTransform(transform).
   *
   * @param transform The transform to apply.
   * @return The robot pose with the new end effector frame.
   */
  [[nodiscard]] RobotPose changeEndEffectorFrame(const Affine &transform) const { return rightTransform(transform); }

  /**
   * @brief Get the pose with a new elbow state.
   *
   * @param elbow_state The new elbow state.
   * @return The pose with the new elbow state.
   */
  [[nodiscard]] RobotPose withElbowState(const std::optional<ElbowState> elbow_state) const {
    return {end_effector_pose_, elbow_state};
  }

  /**
   * @brief Get the end effector pose.
   *
   * @return The end effector pose.
   */
  [[nodiscard]] Affine end_effector_pose() const { return end_effector_pose_; }

  /**
   * @brief Get the elbow state.
   *
   * @return The elbow state.
   */
  [[nodiscard]] std::optional<ElbowState> elbow_state() const { return elbow_state_; }

  friend std::ostream &operator<<(std::ostream &os, const RobotPose &robot_pose);

 private:
  Affine end_effector_pose_;
  std::optional<ElbowState> elbow_state_;
};

inline RobotPose operator*(const RobotPose &robot_pose, const Affine &right_transform) {
  return robot_pose.rightTransform(right_transform);
}

inline RobotPose operator*(const Affine &left_transform, const RobotPose &robot_pose) {
  return robot_pose.leftTransform(left_transform);
}

}  // namespace franky
