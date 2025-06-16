#pragma once

#include <array>
#include <iostream>
#include <optional>

namespace franky {

/**
 * @brief Flip direction of a joint.
 */
enum class FlipDirection { kNegative = -1, kNeutral = 0, kPositive = 1 };

std::ostream &operator<<(std::ostream &os, const FlipDirection &flip_direction);

/**
 * @brief Elbow state of the robot.
 *
 * This class encapsulates the elbow state of a robot, which comprises the
 * position of the 3rd joint and the flip direction of the 4th joint. For
 * details see
 * https://frankaemika.github.io/libfranka/0.15.0/structfranka_1_1RobotState.html#a43485841c427d70e7f36a912cc3116d1
 */
class ElbowState {
 public:
  /**
   * @brief Construct an elbow state with the given joint position and optional
   * flip direction.
   *
   * @param joint_3_pos  The position of the 3rd joint.
   * @param joint_4_flip The flip direction of the 4th joint.
   */
  explicit ElbowState(double joint_3_pos, std::optional<FlipDirection> joint_4_flip = std::nullopt)
      : joint_3_pos_(joint_3_pos), joint_4_flip_(joint_4_flip) {}

  /**
   * @brief Construct an elbow state from an array containing joint position and
   * flip direction.
   *
   * @param elbow_state The joint position and flip direction as a double array.
   */
  explicit ElbowState(const std::array<double, 2> &elbow_state)
      : joint_3_pos_(elbow_state[0]), joint_4_flip_([&]() -> std::optional<FlipDirection> {
          if (elbow_state[1] < 0.0) {
            return FlipDirection::kNegative;
          } else if (elbow_state[1] == 0.0) {
            return FlipDirection::kNeutral;
          } else {
            return FlipDirection::kPositive;
          }
        }()) {}

  ElbowState(const ElbowState &) = default;
  ElbowState() = default;

  /**
   * @brief Get the joint position and flip direction as an array.
   *
   * @param default_flip_direction The default flip direction to use if not
   * explicitly set.
   * @return std::array containing the joint position and flip direction.
   */
  [[nodiscard]] inline std::array<double, 2> to_array(
      FlipDirection default_flip_direction = FlipDirection::kNegative) const {
    return {joint_3_pos_, static_cast<double>(joint_4_flip_.value_or(default_flip_direction))};
  }

  /**
   * @brief The position of the 3rd joint.
   */
  [[nodiscard]] inline double joint_3_pos() const { return joint_3_pos_; }

  /**
   * @brief The flip direction of the 4th joint.
   */
  [[nodiscard]] inline std::optional<FlipDirection> joint_4_flip() const { return joint_4_flip_; }

  friend std::ostream &operator<<(std::ostream &os, const ElbowState &elbow_state);

 private:
  double joint_3_pos_{0.0};
  std::optional<FlipDirection> joint_4_flip_{std::nullopt};
};

}  // namespace franky
