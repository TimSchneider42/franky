#pragma once

#include <functional>
#include <memory>
#include <optional>

#include "franky/motion/cartesian_impedance_base.hpp"
#include "franky/motion/impedance_gains_handle.hpp"
#include "franky/motion/wait_free_triple_buffer.hpp"

namespace franky {

/**
 * @brief Double-buffered handle for updating a CartesianReference online.
 *
 * This handle is intended to be written from a user thread while a single
 * CartesianImpedanceTrackingMotion is running. The motion reads the latest
 * valid reference each control cycle without needing to replace the motion
 * object.
 */
using CartesianReferenceHandle = WaitFreeTripleBuffer<CartesianReference>;

/**
 * @brief Cartesian impedance tracking motion.
 *
 * This motion keeps the same Cartesian impedance controller alive while
 * reading the latest reference from a handle or callback every control cycle.
 */
class CartesianImpedanceTrackingMotion : public CartesianImpedanceBase {
 public:
  using Params = CartesianImpedanceBase::Params;
  using ReferenceCallback =
      std::function<CartesianReference(const RobotState &, franka::Duration, franka::Duration, franka::Duration)>;

  explicit CartesianImpedanceTrackingMotion(std::shared_ptr<CartesianReferenceHandle> reference_handle);
  CartesianImpedanceTrackingMotion(std::shared_ptr<CartesianReferenceHandle> reference_handle, const Params &params);
  CartesianImpedanceTrackingMotion(
      std::shared_ptr<CartesianReferenceHandle> reference_handle, const Params &params,
      std::shared_ptr<CartesianImpedanceGainsHandle> gains_handle, double gains_time_constant = 0.1);
  explicit CartesianImpedanceTrackingMotion(ReferenceCallback reference_callback);
  CartesianImpedanceTrackingMotion(ReferenceCallback reference_callback, const Params &params);

  [[nodiscard]] const Params &params() const { return base_params(); }

 protected:
  void initImpl(const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) override;
  franka::Torques nextCommandImpl(
      const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
      const std::optional<franka::Torques> &previous_command) override;

 private:
  std::shared_ptr<CartesianReferenceHandle> reference_handle_;
  ReferenceCallback reference_callback_;
  std::optional<Twist> target_twist_;
  std::optional<TwistAcceleration> target_acceleration_;
};

}  // namespace franky
