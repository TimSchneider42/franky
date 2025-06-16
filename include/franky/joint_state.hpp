#pragma once

#include <Eigen/Core>

#include "franky/robot_pose.hpp"
#include "franky/robot_velocity.hpp"

namespace franky {

/**
 * @brief Joint state of a robot.
 *
 * This class encapsulates the joint state of a robot, which comprises the joint
 * positions and the joint velocities.
 */
class JointState {
 public:
  // Suppress implicit conversion warning
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-conversion"
  /**
   * @brief Construct a joint state with the given joint positions and zero
   * velocities.
   *
   * @param position The joint positions.
   */
  JointState(Vector7d position) : position_(std::move(position)), velocity_(Vector7d::Zero()) {}
#pragma clang diagnostic pop

  /**
   * @param position The joint positions.
   * @param velocity The joint velocities.
   */
  JointState(Vector7d position, Vector7d velocity) : position_(std::move(position)), velocity_(std::move(velocity)) {}

  JointState(const JointState &) = default;

  JointState() = default;

  /**
   * @brief The position component of the state.
   */
  [[nodiscard]] inline Vector7d position() const { return position_; }

  /**
   * @brief The velocity component of the state.
   */
  [[nodiscard]] inline Vector7d velocity() const { return velocity_; }

  friend std::ostream &operator<<(std::ostream &os, const JointState &joint_state);

 private:
  Vector7d position_;
  Vector7d velocity_;
};

inline std::ostream &operator<<(std::ostream &os, const JointState &joint_state) {
  os << "JointState(position=" << joint_state.position_ << ", velocity=" << joint_state.velocity_ << ")";
  return os;
}

}  // namespace franky
