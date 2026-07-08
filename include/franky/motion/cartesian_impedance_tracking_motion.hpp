#pragma once

#include <functional>
#include <memory>
#include <optional>

#include "franky/motion/cartesian_impedance_base.hpp"
#include "franky/wait_free_triple_buffer.hpp"

namespace franky {

/**
 * @brief Double-buffered handle for updating a CartesianReference online.
 *
 * This handle is intended to be written from a user thread while a single
 * CartesianImpedanceTrackingMotion is running. The motion reads the latest
 * valid reference each control cycle without needing to replace the motion
 * object.
 */
using CartesianReferenceHandle = WaitFreeTripleBuffer<std::optional<CartesianReference>>;

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

  explicit CartesianImpedanceTrackingMotion(const Params &params = Params{}, double gains_time_constant = 0.1);
  explicit CartesianImpedanceTrackingMotion(
      ReferenceCallback reference_callback, const Params &params = Params{}, double gains_time_constant = 0.1);

  void setReference(const CartesianReference &reference) { reference_handle_.set(reference); }
  [[nodiscard]] std::optional<CartesianReference> getReference() const { return reference_handle_.getLastWritten(); }

  [[nodiscard]] const Params &params() const { return base_params(); }

 protected:
  void initImpl(const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) override;
  franka::Torques nextCommandImpl(
      const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
      const std::optional<franka::Torques> &previous_command) override;

 private:
  CartesianReferenceHandle reference_handle_{std::nullopt};
  ReferenceCallback reference_callback_;
  std::optional<Twist> target_twist_;
  std::optional<TwistAcceleration> target_acceleration_;
};

}  // namespace franky
