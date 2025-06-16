#pragma once

#include <franky/robot_state.hpp>
#include <functional>

namespace franky {

/**
 * @brief A condition on the robot state.
 *
 * This class defines a condition on the robot state, which can be used to
 * define a condition for a reaction in a motion. Conditions support logical
 * operations (&&, ||, ==, !=, !) and can be combined to form more complex
 * conditions.
 */
class Condition {
 public:
  using CheckFunc = std::function<bool(const RobotState &, franka::Duration, franka::Duration)>;

  /**
   * @param check_func A function that returns true if the condition is met.
   * @param repr A string representation of the condition.
   */
  explicit Condition(CheckFunc check_func, std::string repr = "NULL");

  /**
   * @brief Implicit constructor for constant conditions.
   * @param constant_value The constant value of the condition.
   */
  Condition(bool constant_value);

  /**
   * @brief Check if the condition is met.
   *
   * @param robot_state The current robot state.
   * @param rel_time The time since the start of the current motion.
   * @param abs_time The time since the start of the current chain of motions.
   * This value measures the time since the robot started moving, and is only
   * reset if a motion expires without being replaced by a new motion.
   * @return True if the condition is met.
   */
  inline bool operator()(const RobotState &robot_state, franka::Duration rel_time, franka::Duration abs_time) const {
    return check_func_(robot_state, rel_time, abs_time);
  }

  /**
   * @brief The string representation of the condition.
   */
  [[nodiscard]] inline std::string repr() const { return repr_; }

 private:
  CheckFunc check_func_;
  std::string repr_;
};

Condition operator&&(const Condition &c1, const Condition &c2);

Condition operator||(const Condition &c1, const Condition &c2);

Condition operator==(const Condition &c1, const Condition &c2);

Condition operator!=(const Condition &c1, const Condition &c2);

Condition operator!(const Condition &c);

}  // namespace franky
