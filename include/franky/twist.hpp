#pragma once

#include <Eigen/Core>

#include "franky/types.hpp"
#include "franky/util.hpp"

namespace franky {

/**
 * @brief Twist of a frame.
 */
class Twist {
 public:
  Twist(const Twist &twist) = default;

  /**
   * @param linear_velocity The linear velocity in [m/s].
   * @param angular_velocity The angular velocity in [rad/s].
   */
  explicit Twist(
      Eigen::Vector3d linear_velocity = Eigen::Vector3d::Zero(),
      Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero())
      : linear_velocity_(std::move(linear_velocity)), angular_velocity_(std::move(angular_velocity)) {}

  /**
   * @param vector_repr The vector representation of the twist.
   */
  [[nodiscard]] static Twist fromVectorRepr(const Vector6d &vector_repr) {
    return Twist{vector_repr.head<3>(), vector_repr.tail<3>()};
  }

  /**
   * @brief Get the vector representation of the twist. It consists of the
   * linear and angular velocities.
   *
   * @return The vector representation of the twist.
   */
  [[nodiscard]] Vector6d vector_repr() const {
    Vector6d result;
    result << linear_velocity_, angular_velocity_;
    return result;
  }

  /**
   * @brief Transform the frame of the twist by applying the given affine
   * transform.
   *
   * @param transformation The transformation to apply.
   * @return The twist after the transformation.
   */
  [[nodiscard]] Twist transformWith(const Affine &transformation) const {
    return transformWith(transformation.rotation());
  }

  /**
   * @brief Transform the frame of the twist by applying the given rotation.
   *
   * @param rotation The rotation to apply.
   * @return The twist after the transformation.
   */
  template <typename RotationMatrixType>
  [[nodiscard]] Twist transformWith(const RotationMatrixType &rotation) const {
    return Twist{rotation * linear_velocity_, rotation * angular_velocity_};
  }

  /**
   * @brief Propagate the twist through a link with the given translation.
   * Hence, suppose this twist is the twist of a frame A, then this function
   * computes the twist of a frame B that is rigidly attached to frame A by a
   * link with the given translation: B = A + T, where T is the translation.
   *
   * @param link_translation: The translation of the link. Must be in the same
   * reference frame as this twist.
   * @return The twist propagated through the link.
   */
  [[nodiscard]] Twist propagateThroughLink(const Eigen::Vector3d &link_translation) const {
    return Twist{linear_velocity_ + angular_velocity_.cross(link_translation), angular_velocity_};
  }

  /**
   * @brief Get the linear velocity.
   *
   * @return The linear velocity [m/s].
   */
  [[nodiscard]] Eigen::Vector3d linear_velocity() const { return linear_velocity_; }

  /**
   * @brief Get the angular velocity.
   *
   * @return The angular velocity [rad/s].
   */
  [[nodiscard]] Eigen::Vector3d angular_velocity() const { return angular_velocity_; }

  friend std::ostream &operator<<(std::ostream &os, const Twist &twist);

 private:
  Eigen::Vector3d linear_velocity_;
  Eigen::Vector3d angular_velocity_;
};

inline Twist operator*(const Affine &affine, const Twist &twist) { return twist.transformWith(affine); }

template <typename RotationMatrixType>
Twist operator*(const RotationMatrixType &rotation, const Twist &twist) {
  return twist.transformWith(rotation);
}

inline std::ostream &operator<<(std::ostream &os, const Twist &twist) {
  os << "Twist(lin=" << twist.linear_velocity_ << ", ang=" << twist.angular_velocity_ << ")";
  return os;
}

}  // namespace franky
