#include "franky/gripper.hpp"

namespace franky {
std::shared_future<bool> Gripper::graspAsync(
    double width, double speed, double force, double epsilon_inner, double epsilon_outer) {
  return setCurrentFuture(
      std::async(std::launch::async, &Gripper::grasp, this, width, speed, force, epsilon_inner, epsilon_outer));
}

std::shared_future<bool> Gripper::moveAsync(double width, double speed) {
  return setCurrentFuture(std::async(std::launch::async, &Gripper::move, this, width, speed));
}

bool Gripper::open(double speed) { return move(max_width(), speed); }

std::shared_future<bool> Gripper::openAsync(double speed) {
  return setCurrentFuture(std::async(std::launch::async, &Gripper::open, this, speed));
}

std::shared_future<bool> Gripper::homingAsync() {
  return setCurrentFuture(std::async(std::launch::async, &Gripper::homing, this));
}

std::shared_future<bool> Gripper::stopAsync() {
  return setCurrentFuture(std::async(std::launch::async, &Gripper::stop, this));
}

std::shared_future<bool> Gripper::setCurrentFuture(std::future<bool> future) {
  if (current_future_.valid()) current_future_.wait();
  current_future_ = future.share();
  return current_future_;
}

}  // namespace franky
