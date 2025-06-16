#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace franky {

/**
 * @brief A template class representing a dynamics limit with a maximum value.
 *
 * This class provides functionality to store and manage a dynamic limit,
 * including setting and getting the value of the limit while ensuring that it
 * cannot be written while the robot is in control.
 *
 * @tparam LimitType The type of the limit value (e.g., double, int).
 */
template <typename LimitType>
class DynamicsLimit {
 public:
  /**
   * @brief Constructor for DynamicsLimit.
   *
   * Initializes a new instance of the DynamicsLimit class with the provided
   * values. The maximum value is used as default value by this constructor.
   *
   * @param desc                A string description of the limit (e.g., "joint
   * velocity").
   * @param max_val             The maximum allowable value for this limit.
   * @param write_mutex         A shared pointer to the mutex used for
   * synchronizing writes to the limit.
   * @param can_write_condition A function that returns true if the limit can be
   * written to.
   */
  DynamicsLimit(
      const std::string &desc, const LimitType &max_val, const std::shared_ptr<std::mutex> &write_mutex,
      const std::function<bool()> &can_write_condition)
      : DynamicsLimit(desc, max_val, write_mutex, can_write_condition, max_val) {}

  /**
   * @brief Constructor for DynamicsLimit.
   *
   * Initializes a new instance of the DynamicsLimit class with the provided
   * values.
   *
   * @param desc                A string description of the limit (e.g., "joint
   * velocity").
   * @param max_val             The maximum allowable value for this limit.
   * @param write_mutex         A shared pointer to the mutex used for
   * synchronizing writes to the limit.
   * @param can_write_condition A function that returns true if the limit can be
   * written to.
   * @param default_val         The default value for this limit.
   */
  DynamicsLimit(
      std::string desc, const LimitType &max_val, const std::shared_ptr<std::mutex> &write_mutex,
      const std::function<bool()> &can_write_condition, const LimitType &default_val)
      : max(max_val),
        desc(std::move(desc)),
        write_mutex_(write_mutex),
        value_(default_val),
        can_write_condition_(can_write_condition) {}

  /**
   * @brief Set a new value for the limit with a different type.
   *
   * This version of `set` allows setting the value with a different type, with
   * appropriate type conversions.
   *
   * @tparam AlternativeType The type of the alternative value to set.
   * @param value The new value to set for the limit.
   */
  template <typename AlternativeType>
  void setFrom(const AlternativeType &value);

  /**
   * @brief Set a new value for the limit.
   *
   * Sets the value of the limit, but only if the robot is not in control.
   * Throws a runtime_error if the condition for writing to the limit is not
   * met.
   *
   * @param value The new value to set for the limit.
   *
   * @throws std::runtime_error if the limit cannot be set due to the robot
   * being in control.
   */
  void set(const LimitType &value) {
    std::unique_lock lock(*write_mutex_);
    if (!can_write_condition_()) {
      std::stringstream ss;
      ss << "Cannot set " << desc << " limit while robot is in control.";
      throw std::runtime_error(ss.str());
    }
    check(value);
    value_ = value;
  }

  /**
   * @brief Get the current value of the limit.
   *
   * Retrieves the current value stored in this limit.
   *
   * @return The current value of the limit.
   */
  [[nodiscard]] LimitType get() const { return value_; }

  /**
   * @brief The maximum value this limit can take as defined by Franka.
   *
   * This value represents the maximum boundary for the limit, beyond which the
   * value cannot be set.
   */
  const LimitType max;

  /**
   * @brief Description of this limit.
   *
   * This string provides a human-readable description of the limit, such as
   * "joint velocity" or "maximum load".
   */
  const std::string desc;

  template <typename LimitTypeStream>
  friend std::ostream &operator<<(std::ostream &os, const DynamicsLimit<LimitTypeStream> &dynamics_limit);

 private:
  /**
   * @brief Validate the new value before setting it.
   *
   * This function checks whether the given value is within acceptable bounds
   * for the limit. It is called before setting the limit to ensure correctness.
   *
   * @param value The value to check.
   */
  void check(const LimitType &value) const;

  std::shared_ptr<std::mutex> write_mutex_;   /**< Mutex for synchronizing writes to the limit. */
  LimitType value_;                           /**< Current value of the limit. */
  std::function<bool()> can_write_condition_; /**< Function to check if writing is allowed. */
};

template <typename LimitTypeStream>
std::ostream &operator<<(std::ostream &os, const DynamicsLimit<LimitTypeStream> &dynamics_limit) {
  os << dynamics_limit.value_ << " (max: " << dynamics_limit.max << ")";
  return os;
}

}  // namespace franky
