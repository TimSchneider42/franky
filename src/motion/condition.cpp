#include "franky/motion/condition.hpp"

#include <utility>

namespace franky {

Condition::Condition(Condition::CheckFunc check_func, std::string repr)
    : check_func_(std::move(check_func)), repr_(std::move(repr)) {}

Condition::Condition(bool constant_value)
    : Condition(
          [constant_value](const RobotState &robot_state, franka::Duration rel_time, franka::Duration abs_time) {
            return constant_value;
          },
          constant_value ? "true" : "false") {}

Condition operator&&(const Condition &c1, const Condition &c2) {
  return Condition(
      [c1, c2](const RobotState &robot_state, franka::Duration rel_time, franka::Duration abs_time) {
        return c1(robot_state, rel_time, abs_time) && c2(robot_state, rel_time, abs_time);
      },
      "(" + c1.repr() + ") && (" + c2.repr() + ")");
}

Condition operator||(const Condition &c1, const Condition &c2) {
  return Condition(
      [c1, c2](const RobotState &robot_state, franka::Duration rel_time, franka::Duration abs_time) {
        return c1(robot_state, rel_time, abs_time) || c2(robot_state, rel_time, abs_time);
      },
      "(" + c1.repr() + ") || (" + c2.repr() + ")");
}

Condition operator==(const Condition &c1, const Condition &c2) {
  return Condition(
      [c1, c2](const RobotState &robot_state, franka::Duration rel_time, franka::Duration abs_time) {
        return c1(robot_state, rel_time, abs_time) == c2(robot_state, rel_time, abs_time);
      },
      c1.repr() + " == " + c2.repr());
}

Condition operator!=(const Condition &c1, const Condition &c2) {
  return Condition(
      [c1, c2](const RobotState &robot_state, franka::Duration rel_time, franka::Duration abs_time) {
        return c1(robot_state, rel_time, abs_time) != c2(robot_state, rel_time, abs_time);
      },
      c1.repr() + " != " + c2.repr());
}

Condition operator!(const Condition &c) {
  return Condition(
      [c](const RobotState &robot_state, franka::Duration rel_time, franka::Duration abs_time) {
        return !c(robot_state, rel_time, abs_time);
      },
      "!(" + c.repr() + ")");
}

}  // namespace franky
