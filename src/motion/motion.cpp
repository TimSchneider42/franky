#include "franky/motion/motion.hpp"

#include <franky/rt_mutex.hpp>
#include <mutex>

#include "franky/motion/reaction.hpp"
#include "franky/robot.hpp"

namespace franky {

template class Motion<franka::Torques>;
template class Motion<franka::JointVelocities>;
template class Motion<franka::CartesianVelocities>;
template class Motion<franka::JointPositions>;
template class Motion<franka::CartesianPose>;

template <typename ControlSignalType>
Motion<ControlSignalType>::Motion() : robot_(nullptr) {
  patchMutexRT(callback_mutex_);
  patchMutexRT(reaction_mutex_);
}

template <typename ControlSignalType>
void Motion<ControlSignalType>::registerCallback(CallbackType callback) {
  const std::lock_guard lock(callback_mutex_);
  callbacks_.push_back(callback);
}

template <typename ControlSignalType>
void Motion<ControlSignalType>::addReaction(const std::shared_ptr<Reaction<ControlSignalType>> reaction) {
  if (reaction == nullptr) throw std::invalid_argument("The reaction must not be null.");
  const std::lock_guard lock(reaction_mutex_);
  reactions_.push_back(reaction);
}

template <typename ControlSignalType>
void Motion<ControlSignalType>::addReactionFront(const std::shared_ptr<Reaction<ControlSignalType>> reaction) {
  if (reaction == nullptr) throw std::invalid_argument("The reaction must not be null.");
  const std::lock_guard lock(reaction_mutex_);
  reactions_.push_front(reaction);
}

template <typename ControlSignalType>
std::vector<std::shared_ptr<Reaction<ControlSignalType>>> Motion<ControlSignalType>::reactions() {
  const std::lock_guard lock(reaction_mutex_);
  return std::vector(reactions_.begin(), reactions_.end());
}

template <typename ControlSignalType>
void Motion<ControlSignalType>::init(
    Robot *robot, const RobotState &robot_state, const std::optional<ControlSignalType> &previous_command) {
  robot_ = robot;
  initImpl(robot_state, previous_command);
}

template <typename ControlSignalType>
ControlSignalType Motion<ControlSignalType>::nextCommand(
    const RobotState &robot_state, franka::Duration time_step, franka::Duration rel_time, franka::Duration abs_time,
    const std::optional<ControlSignalType> &previous_command) {
  auto next_command = nextCommandImpl(robot_state, time_step, rel_time, abs_time, previous_command);
  const std::lock_guard lock(callback_mutex_);
  for (auto const &cb : callbacks_) cb(robot_state, time_step, rel_time, abs_time, next_command);
  return next_command;
}

template <typename ControlSignalType>
std::shared_ptr<Motion<ControlSignalType>> Motion<ControlSignalType>::checkAndCallReactions(
    const RobotState &robot_state, franka::Duration rel_time, franka::Duration abs_time) {
  for (auto &reaction : reactions_) {
    if (reaction->condition(robot_state, rel_time, abs_time)) {
      auto new_motion = (*reaction)(robot_state, rel_time, abs_time);
      if (new_motion != nullptr) return new_motion;
    }
  }
  return nullptr;
}

}  // namespace franky
