#pragma once

#include <string>

namespace franky {

/**
 * @brief Relative dynamics factors
 *
 * This class encapsulates the relative dynamics factors, which are used to
 * scale the maximum velocity, acceleration, and jerk of a trajectory.
 */
class RelativeDynamicsFactor {
 public:
  /**
   * @brief Default constructor which initializes all values to 1.0.
   */
  RelativeDynamicsFactor();

  // Suppress implicit conversion warning
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-conversion"
  /**
   * @brief Constructor which initializes all values to the given value.
   *
   * @param value The value to initialize all factors with. Must be in the range
   * (0, 1].
   */
  RelativeDynamicsFactor(double value);
#pragma clang diagnostic pop

  /**
   * @param velocity The factor for the velocity. Must be in the range (0, 1].
   * @param acceleration The factor for the acceleration. Must be in the range
   * (0, 1].
   * @param jerk The factor for the jerk. Must be in the range (0, 1].
   */
  RelativeDynamicsFactor(double velocity, double acceleration, double jerk);

  /**
   * @brief Special factor which causes the maximum dynamics to be used,
   * independent of other factors applied elsewhere.
   */
  static inline RelativeDynamicsFactor MAX_DYNAMICS() { return {1.0, 1.0, 1.0, true}; }

  /**
   * @brief Velocity factor.
   */
  [[nodiscard]] inline double velocity() const { return velocity_; }

  /**
   * @brief Acceleration factor.
   */
  [[nodiscard]] inline double acceleration() const { return acceleration_; }

  /**
   * @brief Jerk factor.
   */
  [[nodiscard]] inline double jerk() const { return jerk_; }

  /**
   * @brief Whether the maximum dynamics should be used.
   */
  [[nodiscard]] inline bool max_dynamics() const { return max_dynamics_; }

  RelativeDynamicsFactor operator*(const RelativeDynamicsFactor &other) const;

 private:
  static double checkInBounds(double value, const std::string &name);

  RelativeDynamicsFactor(double velocity, double acceleration, double jerk, bool max_dynamics);

  double velocity_, acceleration_, jerk_;
  bool max_dynamics_;
};

}  // namespace franky
