#include "franky/relative_dynamics_factor.hpp"

#include <sstream>
#include <stdexcept>

namespace franky {

RelativeDynamicsFactor::RelativeDynamicsFactor() : RelativeDynamicsFactor(1.0) {}

RelativeDynamicsFactor::RelativeDynamicsFactor(double value) : RelativeDynamicsFactor(value, value, value) {}

RelativeDynamicsFactor::RelativeDynamicsFactor(double velocity, double acceleration, double jerk)
    : RelativeDynamicsFactor(velocity, acceleration, jerk, false) {}

RelativeDynamicsFactor::RelativeDynamicsFactor(double velocity, double acceleration, double jerk, bool max_dynamics)
    : velocity_(checkInBounds(velocity, "velocity")),
      acceleration_(checkInBounds(acceleration, "acceleration")),
      jerk_(checkInBounds(jerk, "jerk")),
      max_dynamics_(max_dynamics) {}

double RelativeDynamicsFactor::checkInBounds(double value, const std::string &name) {
  if (0 < value && value <= 1) return value;
  std::stringstream ss;
  ss << "Value " << value << " is out of bounds (0, 1] for parameter " << name << ".";
  throw std::runtime_error(ss.str());
}

RelativeDynamicsFactor RelativeDynamicsFactor::operator*(const RelativeDynamicsFactor &other) const {
  if (max_dynamics_ || other.max_dynamics_) return MAX_DYNAMICS();
  return {velocity_ * other.velocity_, acceleration_ * other.acceleration_, jerk_ * other.jerk_};
}

}  // namespace franky
