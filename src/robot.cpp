#include "franky/robot.hpp"

#include "franky/rt_mutex.hpp"
#include "franky/types.hpp"
#include "franky/util.hpp"

namespace franky {

namespace {

constexpr uint64_t kRobotStateFreshnessToleranceMS = 3;

[[nodiscard]] bool robotStateIsStale(
    double host_advance_ms, uint64_t previous_robot_time_ms, uint64_t candidate_robot_time_ms) {
  if (candidate_robot_time_ms < previous_robot_time_ms) return true;
  const auto robot_advance_ms = candidate_robot_time_ms - previous_robot_time_ms;
  return host_advance_ms > static_cast<double>(robot_advance_ms + kRobotStateFreshnessToleranceMS);
}

}  // namespace

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
      relative_dynamics_factor_handle_(params.relative_dynamics_factor),
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
  patchMutexRT(*control_mutex_);
  model_ = std::make_shared<const Model>(loadModel());
#ifdef FRANKA_0_15
  model_urdf_ = getRobotModel();
#endif
  setCollisionBehavior(params_.default_torque_threshold, params_.default_force_threshold);
  // Prime the buffer before asynchronous control can make state() rely on it.
  state_buffer_.set(RobotState::from_franka(readOnce()));
  state_received_at_ = std::chrono::steady_clock::now();
}

Robot::~Robot() noexcept {
  bool control_running = false;
  {
    std::lock_guard lock(*control_mutex_);
    control_running = motion_generator_running_;
  }

  if (control_running) {
    try {
      stop();
    } catch (...) {
      // Destruction must continue so an already finishing control thread can
      // still be joined. Any control error is intentionally discarded below.
    }
  }

  try {
    std::unique_lock lock(*control_mutex_);
    joinMotionUnsafe(lock);
  } catch (...) {
    // joinMotionUnsafe joins before rethrowing the stored control exception.
  }
}

bool Robot::hasErrors() { return static_cast<bool>(state().current_errors); }

bool Robot::recoverFromErrors() {
  automaticErrorRecovery();
  return !hasErrors();
}

RobotState Robot::state() {
  RobotState result;
  {
    std::lock_guard control_lock(*control_mutex_);
    if (!is_in_control_unsafe()) {
      const auto previous = state_buffer_.get();
      auto candidate = readOnce();
      auto received_at = std::chrono::steady_clock::now();
      const auto host_advance_ms = std::chrono::duration<double, std::milli>(received_at - state_received_at_).count();

      if (robotStateIsStale(host_advance_ms, previous.time.toMSec(), candidate.time.toMSec())) {
        // If libfranka gave us data that hasn't advanced as much as the host time since the last read
        // (within 3ms), read again to get fresh data.
        candidate = readOnce();
        received_at = std::chrono::steady_clock::now();
      }
      result = RobotState::from_franka(candidate);
      state_buffer_.set(result);
      state_received_at_ = received_at;
    } else {
      result = state_buffer_.get();
      state_received_at_ = std::chrono::steady_clock::now();
    }
  }
  return result;
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

RelativeDynamicsFactor Robot::relative_dynamics_factor() { return relative_dynamics_factor_handle_.getLastWritten(); }

RelativeDynamicsFactor Robot::relative_dynamics_factor_rt() { return relative_dynamics_factor_handle_.get(); }

void Robot::setRelativeDynamicsFactor(const RelativeDynamicsFactor &relative_dynamics_factor) {
  relative_dynamics_factor_handle_.set(relative_dynamics_factor);
}

}  // namespace franky
