#pragma once

#include <cmath>
#include <iostream>
#include <functional>
#include <memory>
#include <optional>
#include <franka/robot_state.h>

#include "franky/motion/motion.hpp"
#include "franky/motion/condition.hpp"

namespace franky {

template<typename ControlSignalType>
class Motion;

/**
 * @brief A reaction that can be attached to a motion.
 *
 * This class defines a reaction in a motion, which can be used to change the motion based on the robot state.
 * Reactions consist of a condition and a motion that replaces the current motion immediately if the condition is met.
 */
template<typename ControlSignalType>
class Reaction {
  using MotionFunc = std::function<std::shared_ptr<Motion<ControlSignalType>>(
      const franka::RobotState &, std::chrono::duration<double>, std::chrono::duration<double>)>;

 public:
  /**
   * @param condition The condition that must be met for the reaction to be executed.
   * @param new_motion The motion that is executed if the condition is met.
   */
  explicit Reaction(const Condition &condition, std::shared_ptr<Motion<ControlSignalType>> new_motion = nullptr);

  /**
   * @param condition The condition that must be met for the reaction to be executed.
   * @param motion_func A function that returns a motion that is executed if the condition is met.
   */
  explicit Reaction(Condition condition, const MotionFunc &motion_func);

  /**
   * @brief Get the new motion if the condition is met.
   * @param robot_state The current robot state.
   * @param rel_time The time since the start of the current motion.
   * @param abs_time The time since the start of the current chain of motions. This value measures the time since the
   * robot started moving, and is only reset if a motion expires without being replaced by a new motion.
   * @return The new motion if the condition is met, or nullptr otherwise.
   */
  std::shared_ptr<Motion<ControlSignalType>> operator()(const franka::RobotState &robot_state,
                                                        std::chrono::duration<double> rel_time,
                                                        std::chrono::duration<double> abs_time);

  /**
   * @brief Check if the condition is met.
   * @param robot_state The current robot state.
   * @param rel_time The time since the start of the current motion.
   * @param abs_time The time since the start of the current chain of motions. This value measures the time since the
   * robot started moving, and is only reset if a motion expires without being replaced by a new motion.
   * @return True if the condition is met, false otherwise.
   */
  [[nodiscard]] inline bool condition(const franka::RobotState &robot_state,
                                      std::chrono::duration<double> rel_time,
                                      std::chrono::duration<double> abs_time) const {
    return condition_(robot_state, rel_time, abs_time);
  }

  /**
   * @brief Register a callback that is called when the condition of this reaction is met.
   * @param callback The callback to register. Callbacks are called with the robot state, the relative time [s] and the
   * absolute time [s].
   */
  void registerCallback(
      std::function<void(
          const franka::RobotState &,
          std::chrono::duration<double>,
          std::chrono::duration<double>)> callback);

 private:
  MotionFunc motion_func_;
  Condition condition_;
  std::mutex callback_mutex_;
  std::vector<std::function<void(
      const franka::RobotState &,
      std::chrono::duration<double>,
      std::chrono::duration<double>)>> callbacks_{};
};

}  // namespace franky
