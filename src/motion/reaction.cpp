#include "franky/motion/reaction.hpp"

#include <franka/robot_state.h>

#include <franky/rt_mutex.hpp>
#include <memory>
#include <utility>

#include "franky/motion/condition.hpp"
#include "franky/motion/motion.hpp"

namespace franky {

template class Reaction<franka::Torques>;
template class Reaction<franka::JointVelocities>;
template class Reaction<franka::CartesianVelocities>;
template class Reaction<franka::JointPositions>;
template class Reaction<franka::CartesianPose>;

template <typename ControlSignalType>
Reaction<ControlSignalType>::Reaction(
    const Condition &condition, const std::shared_ptr<Motion<ControlSignalType>> new_motion)
    : Reaction(condition, [new_motion](const RobotState &, franka::Duration, franka::Duration) { return new_motion; }) {
  if (new_motion == nullptr) throw std::invalid_argument("The new motion must not be null.");
  patchMutexRT(callback_mutex_);
}

template <typename ControlSignalType>
Reaction<ControlSignalType>::Reaction(Condition condition, const Reaction::MotionFunc &motion_func)
    : condition_(std::move(condition)), motion_func_(motion_func) {}

template <typename ControlSignalType>
std::shared_ptr<Motion<ControlSignalType>> Reaction<ControlSignalType>::operator()(
    const RobotState &robot_state, franka::Duration rel_time, franka::Duration abs_time) {
  std::lock_guard lock(callback_mutex_);
  for (const auto &cb : callbacks_) cb(robot_state, rel_time, abs_time);
  return motion_func_(robot_state, rel_time, abs_time);
}

template <typename ControlSignalType>
void Reaction<ControlSignalType>::registerCallback(
    const std::function<void(const RobotState &, franka::Duration, franka::Duration)> &callback) {
  std::lock_guard lock(callback_mutex_);
  callbacks_.push_back(callback);
}

}  // namespace franky
