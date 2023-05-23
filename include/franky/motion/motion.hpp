#pragma once

#include <mutex>

#include "franky/robot/reaction.hpp"
#include "franky/robot.hpp"

namespace franky {
  class Robot;

  template<typename ControlSignalType>
  class Reaction;

  template<typename ControlSignalType>
  class Motion {
  public:
    explicit Motion() : robot_(nullptr) {}

    void addReaction(const std::shared_ptr<Reaction<ControlSignalType>> reaction) {
      const std::lock_guard<std::mutex> lock(mutex_);
      reactions_.push_back(reaction);
    }

    std::vector<std::shared_ptr<Reaction<ControlSignalType>>> reactions() {
      const std::lock_guard<std::mutex> lock(mutex_);
      return reactions_;
    }

  protected:
    void initImpl(const franka::RobotState &robot_state) {};

    virtual ControlSignalType
    nextCommandImpl(const franka::RobotState &robot_state, franka::Duration time_step, double time) = 0;

    Robot *robot() const {
      return robot_;
    }

  private:
    std::vector<std::shared_ptr<Reaction<ControlSignalType>>> reactions_;
    std::mutex mutex_;
    Robot *robot_;

    void initUnsafe(Robot *robot, const franka::RobotState &robot_state) {
      robot_ = robot;
      initImpl(robot_state);
    };

    ControlSignalType
    nextCommandUnsafe(const franka::RobotState &robot_state, franka::Duration time_step, double time) {
      return nextCommandImpl(robot_state, time_step, time);
    };
  };
} // namespace franky