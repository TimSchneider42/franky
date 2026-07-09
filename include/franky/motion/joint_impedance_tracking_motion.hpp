#pragma once

#include <functional>
#include <optional>

#include "franky/motion/joint_impedance_base.hpp"
#include "franky/wait_free_triple_buffer.hpp"

namespace franky {

/**
 * @brief Double-buffered handle for updating a JointReference online.
 *
 * This handle is intended to be written from a user thread while a single
 * JointImpedanceTrackingMotion is running. The motion reads the latest valid
 * reference each control cycle without needing to replace the motion object.
 */
using JointReferenceHandle = WaitFreeTripleBuffer<std::optional<JointReference>>;

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

  explicit JointImpedanceTrackingMotion(const Params &params = Params{}, double gains_time_constant = 0.1);
  explicit JointImpedanceTrackingMotion(
      ReferenceCallback reference_callback, const Params &params = Params{}, double gains_time_constant = 0.1);

  void setReference(const JointReference &reference) {
    reference.validate();
    reference_handle_.set(reference);
  }
  [[nodiscard]] std::optional<JointReference> getReference() const { return reference_handle_.getLastWritten(); }

 protected:
  void initImpl(const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) override;
  franka::Torques nextCommandImpl(
      const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
      const std::optional<franka::Torques> &previous_command) override;

 private:
  JointReferenceHandle reference_handle_{std::nullopt};
  ReferenceCallback reference_callback_;
};

}  // namespace franky
