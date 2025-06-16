#pragma once

#include <Eigen/Core>

#include "franky/twist.hpp"
#include "franky/types.hpp"
#include "franky/util.hpp"

namespace franky {

/**
 * @brief TwistAcceleration acceleration of a frame (2nd derivative of a pose).
 */
class TwistAcceleration {
 public:
  TwistAcceleration(const TwistAcceleration &twist_acceleration_acceleration) = default;

  /**
   * @param linear_acceleration The linear acceleration in [m/s^2].
   * @param angular_acceleration The angular acceleration in [rad/s^2].
   */
  explicit TwistAcceleration(
      Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero(),
      Eigen::Vector3d angular_acceleration = Eigen::Vector3d::Zero())
      : linear_acceleration_(std::move(linear_acceleration)), angular_acceleration_(std::move(angular_acceleration)) {}

  /**
   * @param vector_repr The vector representation of the twist acceleration.
   */
  [[nodiscard]] static TwistAcceleration fromVectorRepr(const Vector6d &vector_repr) {
    return TwistAcceleration{vector_repr.head<3>(), vector_repr.tail<3>()};
  }

  /**
   * @brief Get the vector representation of the twist acceleration. It consists
   * of the linear and angular velocities.
   *
   * @return The vector representation of the twist acceleration.
   */
  [[nodiscard]] Vector6d vector_repr() const {
    Vector6d result;
    result << linear_acceleration_, angular_acceleration_;
    return result;
  }

  /**
   * @brief Transform the frame of the twist acceleration by applying the given
   * affine transform.
   *
   * @param transformation The transformation to apply.
   * @return The twist acceleration after the transformation.
   */
  [[nodiscard]] TwistAcceleration transformWith(const Affine &transformation) const {
    return transformWith(transformation.rotation());
  }

  /**
   * @brief Transform the frame of the twist acceleration by applying the given
   * rotation.
   *
   * @param rotation The rotation to apply.
   * @return The twist acceleration after the transformation.
   */
  template <typename RotationMatrixType>
  [[nodiscard]] TwistAcceleration transformWith(const RotationMatrixType &rotation) const {
    return TwistAcceleration{rotation * linear_acceleration_, rotation * angular_acceleration_};
  }

  /**
   * @brief Propagate the twist acceleration through a link with the given
   * translation. Hence, suppose this twist acceleration is the twist
   * acceleration of a frame A, then this function computes the twist
   * acceleration of a frame B that is rigidly attached to frame A by a link
   * with the given translation: B = A + T, where T is the translation.
   *
   * @param link_translation        The translation of the link (A to B). Must
   * be in the same reference frame as this twist acceleration.
   * @param base_angular_velocity   Angular velocity of the base frame (A)
   * [rad/s]. Must be in the same reference frame as this twist acceleration.
   * @return The twist acceleration propagated through the link.
   */
  [[nodiscard]] TwistAcceleration propagateThroughLink(
      const Eigen::Vector3d &link_translation, const Eigen::Vector3d &base_angular_velocity) const {
    Eigen::Vector3d propagated_linear_acceleration =
        linear_acceleration_ + angular_acceleration_.cross(link_translation) +
        base_angular_velocity.cross(base_angular_velocity.cross(link_translation));

    return TwistAcceleration{propagated_linear_acceleration, angular_acceleration_};
  }

  /**
   * @brief Get the linear acceleration.
   *
   * @return The linear acceleration [m/s^2].
   */
  [[nodiscard]] Eigen::Vector3d linear_acceleration() const { return linear_acceleration_; }

  /**
   * @brief Get the angular acceleration.
   *
   * @return The angular acceleration [rad/s^2].
   */
  [[nodiscard]] Eigen::Vector3d angular_acceleration() const { return angular_acceleration_; }

  friend std::ostream &operator<<(std::ostream &os, const TwistAcceleration &twist_acceleration);

 private:
  Eigen::Vector3d linear_acceleration_;
  Eigen::Vector3d angular_acceleration_;
};

inline TwistAcceleration operator*(const Affine &affine, const TwistAcceleration &twist_acceleration) {
  return twist_acceleration.transformWith(affine);
}

template <typename RotationMatrixType>
TwistAcceleration operator*(const RotationMatrixType &rotation, const TwistAcceleration &twist_acceleration) {
  return twist_acceleration.transformWith(rotation);
}

inline std::ostream &operator<<(std::ostream &os, const TwistAcceleration &twist_acceleration) {
  os << "TwistAcceleration(lin=" << twist_acceleration.linear_acceleration_
     << ", ang=" << twist_acceleration.angular_acceleration_ << ")";
  return os;
}

}  // namespace franky
