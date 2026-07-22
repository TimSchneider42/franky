#pragma once

#include <franka/control_types.h>

#include <cstdint>
#include <optional>
#include <stdexcept>

#include "franky/motion/motion.hpp"
#include "franky/motion/torque_control_utils.hpp"
#include "franky/types.hpp"
#include "franky/wait_free_triple_buffer.hpp"

namespace franky {

/**
 * @brief Thrown when a SimpleTorqueMotion does not receive a new torque signal in time.
 */
struct TorqueSignalTimeoutException : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/**
 * @brief Parameters for SimpleTorqueMotion.
 */
struct SimpleTorqueParams {
  /** Torque applied until the first torque signal arrives [Nm]. */
  Vector7d initial_torque{Vector7d::Zero()};

  /**
   * Maximum duration [s] the motion tolerates without receiving a new torque signal before it
   * terminates with a TorqueSignalTimeoutException. The watchdog is armed when the motion starts,
   * so the initial torque is only held for this duration as well. Set to nullopt to disable the
   * watchdog and hold the last commanded torque indefinitely.
   */
  std::optional<double> signal_timeout{0.05};

  /** Compensate Coriolis forces using the robot model. */
  bool compensate_coriolis{false};

  /** Shared torque safety limits and soft joint-limit repulsion settings. */
  TorqueSafetyParams safety{};

  /** Joint friction compensation settings. */
  FrictionCompensationParams friction{};

  /** @brief Throw std::invalid_argument if any parameter is out of range. */
  void validate() const {
    validateFinite(initial_torque, "initial_torque");
    if (signal_timeout.has_value() && (!std::isfinite(*signal_timeout) || *signal_timeout <= 0.0)) {
      throw std::invalid_argument("signal_timeout must be finite and positive");
    }
    safety.validate();
    friction.validate();
  }
};

/**
 * @brief Direct joint torque control with a signal watchdog.
 *
 * This motion applies the joint torques it is given, without any feedback law on top. Until the
 * first torque signal arrives, it applies SimpleTorqueParams::initial_torque. New torques are
 * published with setTorque from a user thread, analogous to how the references of the impedance
 * motions are updated, and are picked up by the control loop in the next cycle.
 *
 * As the robot is not commanded by any controller in between torque signals, the motion terminates
 * with a TorqueSignalTimeoutException if no new torque arrives within
 * SimpleTorqueParams::signal_timeout (50ms by default). Hence, the torque has to be updated
 * regularly, even if it does not change.
 *
 * Note that the commanded torques are still subject to the torque rate limit
 * (TorqueSafetyParams::max_delta_tau), so large steps are ramped in over multiple cycles.
 */
class SimpleTorqueMotion : public Motion<franka::Torques> {
 public:
  explicit SimpleTorqueMotion(const SimpleTorqueParams &params = SimpleTorqueParams{});

  /**
   * @param initial_torque The torque applied until the first torque signal arrives [Nm].
   * @param signal_timeout Maximum duration [s] without a new torque signal, or nullopt to disable
   * the watchdog.
   */
  explicit SimpleTorqueMotion(const Vector7d &initial_torque, std::optional<double> signal_timeout = 0.05);

  /**
   * @brief Set the torque applied by the motion [Nm].
   *
   * The torque is validated and picked up by the control loop in the next cycle. It also resets the
   * watchdog, so this function has to be called at least every SimpleTorqueParams::signal_timeout
   * seconds while the motion is running.
   * @param torque The new joint torques [Nm].
   */
  void setTorque(const Vector7d &torque) {
    validateFinite(torque, "torque");
    torque_handle_.set({torque, ++write_seq_});
  }

  /**
   * @brief Get a copy of the last published torque [Nm], or the initial torque if no torque has
   * been published yet.
   */
  [[nodiscard]] Vector7d getTorque() const { return torque_handle_.getLastWritten().tau; }

  /**
   * @brief The parameters of the motion.
   */
  [[nodiscard]] const SimpleTorqueParams &params() const { return params_; }

 protected:
  void initImpl(const RobotState &robot_state, const std::optional<franka::Torques> &previous_command) override;

  franka::Torques nextCommandImpl(
      const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
      const std::optional<franka::Torques> &previous_command) override;

 private:
  // Sequence numbers distinguish a newly published torque from the last one being read again.
  struct TorqueCommand {
    Vector7d tau{Vector7d::Zero()};
    uint64_t seq{0};
  };

  SimpleTorqueParams params_;
  WaitFreeTripleBuffer<TorqueCommand> torque_handle_;

  // Owned by the user thread.
  uint64_t write_seq_{0};

  // Owned by the control loop.
  uint64_t last_seq_{0};
  Vector7d current_torque_{Vector7d::Zero()};
  double last_signal_time_{0.0};
};

}  // namespace franky
