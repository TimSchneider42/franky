#pragma once

#include <franka/exception.h>
#include <franka/gripper.h>

#include <future>

namespace franky {

/**
 * @brief Exception thrown by the gripper class.
 */
struct GripperException : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

/**
 * @brief A wrapper around the franka::Gripper class that adds asynchronous
 * functionality.
 */
class Gripper : public franka::Gripper {
 public:
  explicit Gripper(const std::string &franka_address) : franka::Gripper(franka_address) {}

  Gripper(Gripper &&gripper) noexcept : franka::Gripper(std::move(gripper)) {}

  /**
   * @brief Asynchronous variant of the franka::Gripper::grasp function.
   *
   * @param width Size of the object to grasp in [m].
   * @param speed Closing speed in [m/s].
   * @param force Grasping force in [N].
   * @param epsilon_inner Maximum tolerated deviation when the actual grasped
   * width is smaller than the commanded grasp width.
   * @param epsilon_outer Maximum tolerated deviation when the actual grasped
   * width is larger than the commanded grasp width.
   * @return Future that becomes ready when the grasp is finished. Contains true
   * if an object has been grasped, false otherwise.
   */
  std::shared_future<bool> graspAsync(
      double width, double speed, double force, double epsilon_inner = 0.005, double epsilon_outer = 0.005);

  /**
   * @brief Asynchronous variant of the franka::Gripper::move function.
   * @param width Intended opening width in [m].
   * @param speed Speed of the movement in [m/s].
   * @return Future that becomes ready when the movement is finished. Contains
   * true if the movement was successful,
   */
  std::shared_future<bool> moveAsync(double width, double speed);

  /**
   * @brief Opens the gripper fully.
   * @param speed Speed of the movement in [m/s].
   * @return True if the gripper was opened successfully.
   */
  bool open(double speed);

  /**
   * @brief Asynchronous variant of the open function.
   * @param speed Speed of the movement in [m/s].
   * @return Future that becomes ready when the gripper is fully opened.
   * Contains true if the gripper was opened successfully.
   */
  std::shared_future<bool> openAsync(double speed);

  /**
   * @brief Asynchronous variant of the franka::Gripper::homing function.
   * @return A future that becomes ready when the homing is finished. Contains
   * true if the homing was successful.
   */
  std::shared_future<bool> homingAsync();

  /**
   * @brief Asynchronous variant of the franka::Gripper::stop function.
   * @return A future that becomes ready when the stop is finished. Contains
   * true if the stop was successful.
   */
  std::shared_future<bool> stopAsync();

  /**
   * @brief Current opening width of the gripper [m]
   */
  [[nodiscard]] inline double width() const { return state().width; }

  /**
   * @brief Whether the gripper is grasping
   */
  [[nodiscard]] inline bool is_grasped() const { return state().is_grasped; }

  /**
   * @brief Maximum width of the gripper [m]
   */
  [[nodiscard]] inline double max_width() const { return state().max_width; }

  /**
   * @brief Current gripper state.
   */
  [[nodiscard]] inline franka::GripperState state() const { return readOnce(); }

 private:
  std::shared_future<bool> current_future_;

  std::shared_future<bool> setCurrentFuture(std::future<bool> future);
};

}  // namespace franky
