#pragma once

#include <franka/robot_state.h>

#include <functional>
#include <iostream>

#include "franky/motion/condition.hpp"

namespace franky {

/**
 * @brief A measure on the robot state.
 *
 * This class defines a measure on the robot state, which can be used to define
 * a condition for a reaction in a motion. Measures support arithmetic
 * operations (+, -, *, /, ^) and comparisons (==, !=, <, >, <=, >=) and can be
 * combined to form more complex measures.
 */
class Measure {
  using MeasureFunc = std::function<double(const RobotState &, franka::Duration, franka::Duration)>;

 public:
  /**
   * @param measure_func A function that returns the value of the measure.
   * @param repr A string representation of the measure.
   */
  explicit Measure(MeasureFunc measure_func, std::string repr = "NULL");

  /**
   * @brief Implicit constructor for constant measures.
   * @param constant The constant value of the measure.
   */
  Measure(double constant);

  /**
   * @brief Get the value of the measure.
   *
   * @param robot_state The current robot state.
   * @param rel_time The time since the start of the current motion.
   * @param abs_time The time since the start of the current chain of motions.
   * This value measures the time since the robot started moving, and is only
   * reset if a motion expires without being replaced by a new motion.
   * @return The value of the measure.
   */
  inline double operator()(const RobotState &robot_state, franka::Duration rel_time, franka::Duration abs_time) const {
    return measure_func_(robot_state, rel_time, abs_time);
  }

  /**
   * @brief The string representation of the measure.
   */
  [[nodiscard]] inline std::string repr() const { return repr_; }

  /**
   * @brief A measure that returns the linear force on the end-effector in X
   * direction as by the O_F_ext_hat_K component of the robot state.
   */
  static Measure ForceX();

  /**
   * @brief A measure that returns the linear force on the end-effector in Y
   * direction as by the O_F_ext_hat_K component of the robot state.
   */
  static Measure ForceY();

  /**
   * @brief A measure that returns the linear force on the end-effector in Z
   * direction as by the O_F_ext_hat_K component of the robot state.
   */
  static Measure ForceZ();

  /**
   * @brief A measure that returns the relative time since the start of the
   * current motion.
   */
  static Measure RelTime();

  /**
   * @brief A measure that returns the absolute time since the start of the
   * current chain of motions. The absolute time measures the time since the
   * robot started moving, and is only reset if a motion expires without being
   * replaced by a new motion.
   */
  static Measure AbsTime();

 private:
  MeasureFunc measure_func_;
  std::string repr_;
};

Condition operator==(const Measure &m1, const Measure &m2);
Condition operator!=(const Measure &m1, const Measure &m2);
Condition operator<=(const Measure &m1, const Measure &m2);
Condition operator>=(const Measure &m1, const Measure &m2);
Condition operator<(const Measure &m1, const Measure &m2);
Condition operator>(const Measure &m1, const Measure &m2);

Measure operator+(const Measure &m1, const Measure &m2);
Measure operator-(const Measure &m1, const Measure &m2);
Measure operator*(const Measure &m1, const Measure &m2);
Measure operator/(const Measure &m1, const Measure &m2);

Measure operator-(const Measure &m);

Measure measure_pow(const Measure &base, const Measure &exponent);

}  // namespace franky
