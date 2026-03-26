#pragma once

#include <array>
#include <atomic>
#include <functional>
#include <memory>

#include "franky/motion/cartesian_impedance_base.hpp"

namespace franky {

/**
 * @brief Double-buffered handle for updating a CartesianReference online.
 *
 * This handle is intended to be written from a user thread while a single
 * CartesianImpedanceTrackingMotion is running. The motion reads the latest
 * valid reference each control cycle without needing to replace the motion
 * object.
 */
class CartesianReferenceHandle {
 public:
  CartesianReferenceHandle() = default;

  void set(const CartesianReference &reference);
  void clear();
  [[nodiscard]] bool hasReference() const;
  [[nodiscard]] CartesianReference get() const;

 private:
  std::array<CartesianReference, 2> buffers_{};
  std::atomic<uint8_t> active_index_{0};
  std::atomic<bool> valid_{false};
};

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
  explicit CartesianImpedanceTrackingMotion(ReferenceCallback reference_callback);
  CartesianImpedanceTrackingMotion(ReferenceCallback reference_callback, const Params &params);

 protected:
  void initImpl(const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) override;
  std::tuple<CartesianReference, bool> update(
      const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time) override;

 private:
  std::shared_ptr<CartesianReferenceHandle> reference_handle_;
  ReferenceCallback reference_callback_;
  Affine target_;
  std::optional<Twist> target_twist_;
};

}  // namespace franky
