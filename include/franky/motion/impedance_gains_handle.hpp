#pragma once

#include <array>
#include <atomic>
#include <cmath>

#include "franky/motion/double_buffered_handle.hpp"
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
  double translational_stiffness{2000.0};
  double rotational_stiffness{200.0};
  double nullspace_stiffness{0.0};
};

/**
 * @brief Double-buffered handle for updating Cartesian impedance gains online.
 *
 * Same lock-free pattern as CartesianReferenceHandle. The RT loop reads the
 * target gains each cycle and exponentially interpolates toward them, so
 * stiffness changes are smooth rather than instantaneous.
 */
using CartesianImpedanceGainsHandle = DoubleBufferedHandle<CartesianImpedanceGains>;

/**
 * @brief Target gains for a joint impedance controller.
 */
struct JointImpedanceGains {
  Vector7d stiffness{defaultJointImpedanceStiffness()};
  Vector7d damping{defaultJointImpedanceDamping()};
};

/**
 * @brief Double-buffered handle for updating joint impedance gains online.
 */
using JointImpedanceGainsHandle = DoubleBufferedHandle<JointImpedanceGains>;

}  // namespace franky
