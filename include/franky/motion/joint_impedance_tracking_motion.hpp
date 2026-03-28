#pragma once

#include <array>
#include <atomic>
#include <functional>
#include <memory>

#include "franky/motion/impedance_gains_handle.hpp"
#include "franky/motion/joint_impedance_motion.hpp"

namespace franky {

/**
 * @brief Dynamic joint-space reference for JointImpedanceTrackingMotion.
 *
 * The impedance controller tracks the joint position and velocity reference and
 * adds the optional per-cycle feedforward torque term to the commanded torques.
 */
struct JointReference {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Vector7d q{Vector7d::Zero()};
  Vector7d dq{Vector7d::Zero()};
  Vector7d tau_ff{Vector7d::Zero()};
};

/**
 * @brief Double-buffered handle for updating a JointReference online.
 *
 * This handle is intended to be written from a user thread while a single
 * JointImpedanceTrackingMotion is running. The motion reads the latest valid
 * reference each control cycle without needing to replace the motion object.
 */
class JointReferenceHandle {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  JointReferenceHandle() = default;

  /**
   * @brief Publish a new joint-space reference for the running motion.
   */
  void set(const JointReference &reference);

  /**
   * @brief Mark the handle as having no externally supplied reference.
   */
  void clear();

  /**
   * @brief Whether a valid reference is currently available.
   */
  [[nodiscard]] bool hasReference() const;

  /**
   * @brief Get the most recently published reference.
   */
  [[nodiscard]] JointReference get() const;

 private:
  std::array<JointReference, 2> buffers_{};
  std::atomic<uint8_t> active_index_{0};
  std::atomic<bool> valid_{false};
};

/**
 * @brief Client-side joint impedance controller with a dynamic online reference.
 *
 * This motion keeps the same controller alive while reading the latest valid
 * joint reference from a handle or callback each control cycle.
 */
class JointImpedanceTrackingMotion : public JointImpedanceBase {
 public:
  using Params = JointImpedanceParams;
  using ReferenceCallback =
      std::function<JointReference(const RobotState &, franka::Duration, franka::Duration, franka::Duration)>;

  explicit JointImpedanceTrackingMotion(std::shared_ptr<JointReferenceHandle> reference_handle);
  JointImpedanceTrackingMotion(std::shared_ptr<JointReferenceHandle> reference_handle, const Params &params);
  JointImpedanceTrackingMotion(
      std::shared_ptr<JointReferenceHandle> reference_handle, const Params &params,
      std::shared_ptr<JointImpedanceGainsHandle> gains_handle, double gains_time_constant = 0.1);
  explicit JointImpedanceTrackingMotion(ReferenceCallback reference_callback);
  JointImpedanceTrackingMotion(ReferenceCallback reference_callback, const Params &params);

 protected:
  void initImpl(const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) override;
  franka::Torques nextCommandImpl(
      const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
      const std::optional<franka::Torques> &previous_command) override;

 private:
  std::shared_ptr<JointReferenceHandle> reference_handle_;
  ReferenceCallback reference_callback_;
};

}  // namespace franky
