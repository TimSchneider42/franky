#include "franky/robot.hpp"

#include <franky/rt_mutex.hpp>

#include "franky/types.hpp"
#include "franky/util.hpp"

namespace franky {

#ifdef FRANKA_0_10
#define SEL_VAL(value_panda, value_fer) value_fer
#else
#define SEL_VAL(value_panda, value_fer) value_panda
#endif

#define LIMIT_INIT(name, value_panda, value_fer) \
  name, SEL_VAL(value_panda, value_fer), control_mutex_, [this] { return !is_in_control_unsafe(); }

//! Connects to a robot at the given FCI IP address.
Robot::Robot(const std::string &fci_hostname) : Robot(fci_hostname, Params()) {}

Robot::Robot(const std::string &fci_hostname, const Params &params)
    : fci_hostname_(fci_hostname),
      params_(params),
      control_mutex_(std::make_shared<std::mutex>()),
      translation_velocity_limit(LIMIT_INIT("translational velocity", 3.0, 1.7)),
      rotation_velocity_limit(LIMIT_INIT("rotational velocity", 2.5, 2.5)),
      elbow_velocity_limit(LIMIT_INIT("elbow velocity", 2.62, 2.1750)),
      translation_acceleration_limit(LIMIT_INIT("translational acceleration", 9.0, 13.0)),
      rotation_acceleration_limit(LIMIT_INIT("rotational acceleration", 17.0, 25.0)),
      elbow_acceleration_limit(LIMIT_INIT("elbow acceleration", 10.0, 10.0)),
      translation_jerk_limit(LIMIT_INIT("translational jerk", 4500.0, 6500.0)),
      rotation_jerk_limit(LIMIT_INIT("rotational jerk", 8500.0, 12500.0)),
      elbow_jerk_limit(LIMIT_INIT("elbow jerk", 5000.0, 5000.0)),
      joint_velocity_limit(LIMIT_INIT(
          "joint_velocity", toEigenD<7>({2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26}),
          toEigenD<7>({2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61}))),
      joint_acceleration_limit(LIMIT_INIT(
          "joint_acceleration", toEigenD<7>({10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}),
          toEigenD<7>({15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0}))),
      joint_jerk_limit(LIMIT_INIT(
          "joint_jerk", toEigenD<7>({5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0}),
          toEigenD<7>({7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0}))),
      franka::Robot(fci_hostname, params.realtime_config) {
  patchMutexRT(state_mutex_);
  patchMutexRT(*control_mutex_);
  model_ = std::make_shared<const Model>(loadModel());
#ifdef FRANKA_0_15
  model_urdf_ = getRobotModel();
#endif
  setCollisionBehavior(params_.default_torque_threshold, params_.default_force_threshold);
}

bool Robot::hasErrors() { return static_cast<bool>(state().current_errors); }

bool Robot::recoverFromErrors() {
  automaticErrorRecovery();
  return !hasErrors();
}

RobotState Robot::state() {
  std::lock_guard state_lock(state_mutex_);
  {
    std::lock_guard control_lock(*control_mutex_);
    if (!is_in_control_unsafe()) {
      current_state_ = RobotState::from_franka(readOnce());
    }
  }
  return current_state_;
}

void Robot::setCollisionBehavior(const ScalarOrArray<7> &torque_threshold, const ScalarOrArray<6> &force_threshold) {
  setCollisionBehavior(torque_threshold, torque_threshold, force_threshold, force_threshold);
}

void Robot::setCollisionBehavior(
    const ScalarOrArray<7> &lower_torque_threshold, const ScalarOrArray<7> &upper_torque_threshold,
    const ScalarOrArray<6> &lower_force_threshold, const ScalarOrArray<6> &upper_force_threshold) {
  franka::Robot::setCollisionBehavior(
      expand<7>(lower_torque_threshold),
      expand<7>(upper_torque_threshold),
      expand<6>(lower_force_threshold),
      expand<6>(upper_force_threshold));
}

void Robot::setCollisionBehavior(
    const ScalarOrArray<7> &lower_torque_threshold_acceleration,
    const ScalarOrArray<7> &upper_torque_threshold_acceleration, const ScalarOrArray<7> &lower_torque_threshold_nominal,
    const ScalarOrArray<7> &upper_torque_threshold_nominal, const ScalarOrArray<6> &lower_force_threshold_acceleration,
    const ScalarOrArray<6> &upper_force_threshold_acceleration, const ScalarOrArray<6> &lower_force_threshold_nominal,
    const ScalarOrArray<6> &upper_force_threshold_nominal) {
  franka::Robot::setCollisionBehavior(
      expand<7>(lower_torque_threshold_acceleration),
      expand<7>(upper_torque_threshold_acceleration),
      expand<7>(lower_torque_threshold_nominal),
      expand<7>(upper_torque_threshold_nominal),
      expand<6>(lower_force_threshold_acceleration),
      expand<6>(upper_force_threshold_acceleration),
      expand<6>(lower_force_threshold_nominal),
      expand<6>(upper_force_threshold_nominal));
}

bool Robot::is_in_control_unsafe() const { return motion_generator_running_; }

bool Robot::is_in_control() {
  std::unique_lock lock(*control_mutex_);
  return is_in_control_unsafe();
}

std::string Robot::fci_hostname() const { return fci_hostname_; }

std::optional<ControlSignalType> Robot::current_control_signal_type() {
  std::unique_lock lock(*control_mutex_);
  if (!is_in_control_unsafe()) return std::nullopt;
  if (std::holds_alternative<MotionGenerator<franka::Torques>>(motion_generator_)) return Torques;
  if (std::holds_alternative<MotionGenerator<franka::JointVelocities>>(motion_generator_)) return JointVelocities;
  if (std::holds_alternative<MotionGenerator<franka::JointPositions>>(motion_generator_)) return JointPositions;
  if (std::holds_alternative<MotionGenerator<franka::CartesianVelocities>>(motion_generator_))
    return CartesianVelocities;
  return CartesianPose;
}

RelativeDynamicsFactor Robot::relative_dynamics_factor() {
  std::unique_lock lock(*control_mutex_);
  return params_.relative_dynamics_factor;
}

void Robot::setRelativeDynamicsFactor(const RelativeDynamicsFactor &relative_dynamics_factor) {
  std::unique_lock lock(*control_mutex_);
  params_.relative_dynamics_factor = relative_dynamics_factor;
}

}  // namespace franky
