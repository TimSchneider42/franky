#pragma once

#include <Eigen/Core>
#include <optional>

#include "franky/robot_pose.hpp"
#include "franky/robot_velocity.hpp"

namespace franky {

/*
 * @brief Cartesian state of a robot.
 *
 * This class encapsulates the cartesian state of a robot, which comprises the
 * end effector pose and the end effector velocity.
 */
class CartesianState {
 public:
  // Suppress implicit conversion warning
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-conversion"
  CartesianState(const RobotPose &pose) : pose_(pose) {}
#pragma clang diagnostic pop

  /**
   * @param pose The pose of the end effector.
   * @param velocity The velocity of the end effector.
   */
  CartesianState(const RobotPose &pose, const RobotVelocity &velocity) : pose_(pose), velocity_(velocity) {}

  CartesianState(const CartesianState &) = default;

  CartesianState() = default;

  /**
   * @brief Transform the frame of the state by applying the given affine
   * transform.
   *
   * @param transform The transformation to apply.
   * @return The state after the transformation.
   */
  [[nodiscard]] CartesianState transformWith(const Affine &transform) const {
    return {transform * pose_, transform * velocity_};
  }

  /**
   * @brief Change the end effector frame of the state by the given affine
   * transform.
   *
   * @param transform The pose of the new end-effector in the frame of the old
   * end-effector.
   * @return The state with a new end-effector frame.
   */
  [[nodiscard]] CartesianState changeEndEffectorFrame(const Affine &transform) const {
    auto offset_world_frame = pose_.end_effector_pose() * transform.translation();
    return {pose_.changeEndEffectorFrame(transform), velocity_.changeEndEffectorFrame(offset_world_frame)};
  }

  /*
   * @brief Pose component of the state.
   */
  [[nodiscard]] RobotPose pose() const { return pose_; }

  /*
   * @brief Velocity component of the state.
   */
  [[nodiscard]] RobotVelocity velocity() const { return velocity_; }

  friend inline std::ostream &operator<<(std::ostream &os, const CartesianState &cartesian_state);

 private:
  RobotPose pose_;
  RobotVelocity velocity_;
};

inline CartesianState operator*(const Affine &transform, const CartesianState &cartesian_state) {
  return cartesian_state.transformWith(transform);
}

inline std::ostream &operator<<(std::ostream &os, const CartesianState &cartesian_state) {
  os << "CartesianState(pose=" << cartesian_state.pose_ << ", velocity=" << cartesian_state.velocity_ << ")";
  return os;
}

}  // namespace franky
