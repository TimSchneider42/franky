#pragma once

#include <array>
#include <atomic>
#include <cmath>
#include <optional>

#include "franky/motion/torque_control_utils.hpp"
#include "franky/motion/wait_free_triple_buffer.hpp"
#include "franky/types.hpp"

namespace franky {

inline Vector7d defaultJointImpedanceStiffness() { return Vector7d::Constant(50.0); }

inline Vector7d defaultJointImpedanceDamping(const Vector7d &stiffness) { return 2.0 * stiffness.cwiseSqrt(); }

inline Vector7d defaultJointImpedanceDamping() {
  return defaultJointImpedanceDamping(defaultJointImpedanceStiffness());
}

/**
 * @brief Target gains for a Cartesian impedance controller.
 */
struct CartesianImpedanceGains {
  CartesianImpedanceGains() = default;

  explicit CartesianImpedanceGains(
      double translational_stiffness, double rotational_stiffness, double nullspace_stiffness = 0.0)
      : translational_stiffness(translational_stiffness),
        rotational_stiffness(rotational_stiffness),
        nullspace_stiffness(nullspace_stiffness) {
    validate();
  }

  double translational_stiffness{500.0};
  double rotational_stiffness{50.0};
  double nullspace_stiffness{0.0};

  /** @brief Throw std::invalid_argument if any gain is negative or non-finite. */
  void validate() const {
    validateNonNegativeFinite(translational_stiffness, "translational_stiffness");
    validateNonNegativeFinite(rotational_stiffness, "rotational_stiffness");
    validateNonNegativeFinite(nullspace_stiffness, "nullspace_stiffness");
  }
};

/**
 * @brief Double-buffered handle for updating Cartesian impedance gains online.
 *
 * Same lock-free pattern as CartesianReferenceHandle. The RT loop reads the
 * target gains each cycle and exponentially interpolates toward them, so
 * stiffness changes are smooth rather than instantaneous.
 */
using CartesianImpedanceGainsHandle = WaitFreeTripleBuffer<CartesianImpedanceGains>;

/**
 * @brief Target gains for a joint impedance controller.
 */
struct JointImpedanceGains {
  JointImpedanceGains() = default;

  /**
   * @brief Construct gains, defaulting damping to critical damping w.r.t. the stiffness when not provided.
   */
  explicit JointImpedanceGains(
      const std::optional<Vector7d> &stiffness, const std::optional<Vector7d> &damping = std::nullopt)
      : stiffness(stiffness.value_or(defaultJointImpedanceStiffness())),
        damping(damping.has_value() ? *damping : defaultJointImpedanceDamping(this->stiffness)) {
    validate();
  }

  Vector7d stiffness{defaultJointImpedanceStiffness()};
  Vector7d damping{defaultJointImpedanceDamping()};

  /** @brief Throw std::invalid_argument if any gain is negative or non-finite. */
  void validate() const {
    validateNonNegativeFinite(stiffness, "stiffness");
    validateNonNegativeFinite(damping, "damping");
  }
};

/**
 * @brief Double-buffered handle for updating joint impedance gains online.
 */
using JointImpedanceGainsHandle = WaitFreeTripleBuffer<JointImpedanceGains>;

}  // namespace franky
