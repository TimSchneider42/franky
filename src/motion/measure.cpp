#include "franky/motion/measure.hpp"

#include <franka/robot_state.h>

#include <sstream>
#include <utility>

#define MEASURE_CMP_DEF(OP)                                                                             \
  Condition operator OP(const Measure &m1, const Measure &m2) {                                         \
    std::stringstream ss;                                                                               \
    ss << m1.repr() << " " << #OP << " " << m2.repr();                                                  \
    return Condition(                                                                                   \
        [m1, m2](const RobotState &robot_state, franka::Duration rel_time, franka::Duration abs_time) { \
          return m1(robot_state, rel_time, abs_time) OP m2(robot_state, rel_time, abs_time);            \
        },                                                                                              \
        ss.str());                                                                                      \
  }

#define MEASURE_OP_DEF(OP)                                                                              \
  Measure operator OP(const Measure &m1, const Measure &m2) {                                           \
    std::stringstream ss;                                                                               \
    ss << "(" << m1.repr() << ") " << #OP << " (" << m2.repr() << ")";                                  \
    return Measure(                                                                                     \
        [m1, m2](const RobotState &robot_state, franka::Duration rel_time, franka::Duration abs_time) { \
          return m1(robot_state, rel_time, abs_time) OP m2(robot_state, rel_time, abs_time);            \
        },                                                                                              \
        ss.str());                                                                                      \
  }

namespace franky {

Measure::Measure(Measure::MeasureFunc measure_func, std::string repr)
    : measure_func_(std::move(measure_func)), repr_(std::move(repr)) {}

Measure::Measure(double constant)
    : measure_func_([constant](const RobotState &robot_state, franka::Duration rel_time, franka::Duration abs_time) {
        return constant;
      }),
      repr_(std::to_string(constant)) {}

Measure Measure::ForceX() {
  return Measure(
      [](const RobotState &robot_state, franka::Duration rel_time, franka::Duration abs_time) {
        return robot_state.O_F_ext_hat_K[0];
      },
      "F_x");
}

Measure Measure::ForceY() {
  return Measure(
      [](const RobotState &robot_state, franka::Duration rel_time, franka::Duration abs_time) {
        return robot_state.O_F_ext_hat_K[1];
      },
      "F_y");
}

Measure Measure::ForceZ() {
  return Measure(
      [](const RobotState &robot_state, franka::Duration rel_time, franka::Duration abs_time) {
        return robot_state.O_F_ext_hat_K[2];
      },
      "F_z");
}

Measure Measure::RelTime() {
  return Measure(
      [](const RobotState &robot_state, franka::Duration rel_time, franka::Duration abs_time) {
        return rel_time.toSec();
      },
      "t");
}

Measure Measure::AbsTime() {
  return Measure(
      [](const RobotState &robot_state, franka::Duration rel_time, franka::Duration abs_time) {
        return abs_time.toSec();
      },
      "t");
}

MEASURE_CMP_DEF(==)
MEASURE_CMP_DEF(!=)
MEASURE_CMP_DEF(<=)
MEASURE_CMP_DEF(>=)
MEASURE_CMP_DEF(<)
MEASURE_CMP_DEF(>)

MEASURE_OP_DEF(+)
MEASURE_OP_DEF(-)
MEASURE_OP_DEF(*)
MEASURE_OP_DEF(/)

Measure measure_pow(const Measure &base, const Measure &exponent) {
  std::stringstream ss;
  ss << "(" << base.repr() << ")^(" << exponent.repr() << ")";
  return Measure(
      [base, exponent](const RobotState &robot_state, franka::Duration rel_time, franka::Duration abs_time) {
        return std::pow(base(robot_state, rel_time, abs_time), exponent(robot_state, rel_time, abs_time));
      },
      ss.str());
}

Measure operator-(const Measure &m) {
  std::stringstream ss;
  ss << "-(" << m.repr() << ") ";
  return Measure(
      [m](const RobotState &robot_state, franka::Duration rel_time, franka::Duration abs_time) {
        return -m(robot_state, rel_time, abs_time);
      },
      ss.str());
}

}  // namespace franky
