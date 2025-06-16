#pragma once

#include <franka/duration.h>
#include <franka/robot_state.h>

#include <memory>
#include <mutex>

#include "franky/robot_state.hpp"
#include "franky/types.hpp"
#include "ruckig/ruckig.hpp"

namespace franky {

/**
 * @brief Exception thrown when the reaction recursion limit (8) is reached.
 */
struct ReactionRecursionException : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

class Robot;

template <typename ControlSignalType>
class Motion;

/**
 * @brief Helper class for handling motions and reactions.
 */
template <typename ControlSignalType>
class MotionGenerator {
 public:
  /**
   * @brief Maximum recursion limit for reactions. After more than this number
   * of reactions are fired in a single step, the motion generator will throw a
   * ReactionRecursionException.
   */
  static constexpr size_t REACTION_RECURSION_LIMIT = 16;

  /**
   * @param robot The robot instance.
   * @param initial_motion The initial motion.
   */
  explicit MotionGenerator(Robot *robot, std::shared_ptr<Motion<ControlSignalType>> initial_motion);

  /**
   * @brief Update the motion generator and get the next control signal.
   * @param robot_state The current robot state.
   * @param period The time step.
   * @return The control signal for the robot.
   */
  ControlSignalType operator()(const RobotState &robot_state, franka::Duration period);

  /**
   * @brief Register a callback that is called in every step of the motion.
   * @param callback The callback to register. Callbacks are called with the
   * robot state, the time step [s], the relative time [s] and the control
   * signal computed in this step.
   */
  inline void registerUpdateCallback(
      const std::function<void(const RobotState &, franka::Duration, franka::Duration)> &callback) {
    update_callbacks_.push_back(callback);
  }

  /**
   * @brief Update the current motion.
   * @param new_motion The new motion.
   */
  void updateMotion(std::shared_ptr<Motion<ControlSignalType>> new_motion);

  /**
   * @brief Whether a new motion is available.
   */
  bool has_new_motion();

  /**
   * @brief Reset the time of the motion generator without locking the mutex.
   */
  void resetTimeUnsafe();

 private:
  std::shared_ptr<Motion<ControlSignalType>> initial_motion_;
  std::shared_ptr<Motion<ControlSignalType>> current_motion_;
  std::shared_ptr<Motion<ControlSignalType>> new_motion_;
  std::vector<std::function<void(const RobotState &, franka::Duration, franka::Duration)>> update_callbacks_;
  std::mutex new_motion_mutex_;
  std::optional<ControlSignalType> previous_command_;

  franka::Duration abs_time_{0};
  franka::Duration rel_time_offset_{0};
  Robot *robot_;
};

}  // namespace franky
