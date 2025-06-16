#pragma once

#include <list>
#include <mutex>

#include "franky/robot.hpp"
#include "franky/robot_state.hpp"
#include "reaction.hpp"

namespace franky {

class Robot;

template <typename ControlSignalType>
class Reaction;

/**
 * @brief Base class for motions.
 * @tparam ControlSignalType Control signal type of the motion. Either
 * franka::Torques, franka::JointVelocities, franka::CartesianVelocities,
 * franka::JointPositions or franka::CartesianPose.
 */
template <typename ControlSignalType>
class Motion {
 public:
  virtual ~Motion() = default;
  using CallbackType = std::function<void(
      const RobotState &, franka::Duration, franka::Duration, franka::Duration, const ControlSignalType &)>;

  /**
   * @brief Add a reaction to the motion.
   *
   * Reactions are evaluated in every step of the motion and can replace the
   * current motion with a new motion.
   * @param reaction The reaction to add.
   */
  void addReaction(std::shared_ptr<Reaction<ControlSignalType>> reaction);

  /**
   * @brief Add a reaction to the front of the reaction list.
   *
   * Reactions are evaluated in every step of the motion and can replace the
   * current motion with a new motion.
   * @param reaction The reaction to add.
   */
  void addReactionFront(std::shared_ptr<Reaction<ControlSignalType>> reaction);

  /**
   * @brief Currently registered reactions of the motion.
   */
  std::vector<std::shared_ptr<Reaction<ControlSignalType>>> reactions();

  /**
   * @brief Register a callback that is called in every step of the motion.
   * @param callback The callback to register. Callbacks are called with the
   * robot state, the time step [s], the relative time [s], the absolute time
   * [s] and the control signal computed in this step.
   */
  void registerCallback(CallbackType callback);

  /**
   * @brief Initialize the motion.
   * @param robot The robot instance.
   * @param robot_state The current robot state.
   * @param previous_command The previous command.
   */
  void init(Robot *robot, const RobotState &robot_state, const std::optional<ControlSignalType> &previous_command);

  /**
   * @brief Get the next command of the motion.
   * @param robot_state The current robot state.
   * @param time_step The time step [s].
   * @param rel_time The relative time.
   * @param abs_time The absolute time [s].
   * @param previous_command The previous command.
   * @return The next control signal for libfranka.
   */
  ControlSignalType nextCommand(
      const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
      const std::optional<ControlSignalType> &previous_command);

  /**
   * @brief Check and call reactions.
   * @param robot_state The current robot state.
   * @param rel_time The relative time.
   * @param abs_time The absolute time [s].
   * @return The new motion if a reaction was triggered, nullptr otherwise.
   */
  std::shared_ptr<Motion<ControlSignalType>> checkAndCallReactions(
      const RobotState &robot_state, franka::Duration rel_time, franka::Duration abs_time);

 protected:
  explicit Motion();

  virtual void initImpl(const RobotState &robot_state, const std::optional<ControlSignalType> &previous_command) {}

  virtual ControlSignalType nextCommandImpl(
      const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
      const std::optional<ControlSignalType> &previous_command) = 0;

  [[nodiscard]] Robot *robot() const { return robot_; }

 private:
  std::mutex reaction_mutex_;
  std::list<std::shared_ptr<Reaction<ControlSignalType>>> reactions_;
  std::mutex callback_mutex_;
  std::vector<CallbackType> callbacks_;
  Robot *robot_;
};

}  // namespace franky
