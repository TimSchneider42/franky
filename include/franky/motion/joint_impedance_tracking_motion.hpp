#pragma once

#include <functional>
#include <memory>
#include <optional>

#include "franky/motion/double_buffered_handle.hpp"
#include "franky/motion/impedance_gains_handle.hpp"
#include "franky/motion/joint_impedance_base.hpp"

namespace franky {

/**
 * @brief Double-buffered handle for updating a JointReference online.
 *
 * This handle is intended to be written from a user thread while a single
 * JointImpedanceTrackingMotion is running. The motion reads the latest valid
 * reference each control cycle without needing to replace the motion object.
 */
using JointReferenceHandle = DoubleBufferedHandle<JointReference>;

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
