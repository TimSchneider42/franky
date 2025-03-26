#include "franky/robot_velocity.hpp"

#include <optional>
#include <Eigen/Core>
#include <utility>
#include <franka/control_types.h>

#include "franky/types.hpp"
#include "franky/util.hpp"

namespace franky {

RobotVelocity::RobotVelocity() = default;

RobotVelocity::RobotVelocity(const RobotVelocity &) = default;

RobotVelocity::RobotVelocity(const Twist &end_effector_twist, std::optional<double> elbow_velocity)
    : end_effector_twist_(end_effector_twist),
      elbow_velocity_(elbow_velocity) {}

RobotVelocity::RobotVelocity(const Vector7d &vector_repr, bool ignore_elbow)
    : RobotVelocity(Twist::fromVectorRepr(vector_repr.head<6>()),
                    ignore_elbow ? std::optional<double>(std::nullopt) : vector_repr[6]) {}

RobotVelocity::RobotVelocity(const Vector6d &vector_repr, std::optional<double> elbow_velocity)
    : elbow_velocity_(elbow_velocity),
      end_effector_twist_(Twist::fromVectorRepr(vector_repr)) {}

RobotVelocity::RobotVelocity(const franka::CartesianVelocities franka_velocity)
    : RobotVelocity(
    Twist{
        Vector6d::Map(franka_velocity.O_dP_EE.data()).head<3>(),
        Vector6d::Map(franka_velocity.O_dP_EE.data()).tail<3>()
    }, franka_velocity.elbow[0]) {}

Vector7d RobotVelocity::vector_repr() const {
  Vector7d result;
  result << end_effector_twist_.vector_repr(), elbow_velocity_.value_or(0.0);
  return result;
}

franka::CartesianVelocities RobotVelocity::as_franka_velocity(std::optional<double> elbow_position) const {
  std::array<double, 6> array = toStd<6>(vector_repr().head<6>());
  if (elbow_position.has_value())
    return franka::CartesianVelocities(array, {elbow_position.value(), -1});
  return {array};
}

}  // namespace franky
