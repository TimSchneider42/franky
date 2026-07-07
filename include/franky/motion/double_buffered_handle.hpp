#pragma once

#include <array>
#include <atomic>

namespace franky {

/**
 * @brief Double-buffered generic handle for updating data online.
 *
 * This handle is intended to be written from a user thread while a single
 * real-time loop is running. The RT loop reads the latest valid data without
 * locking.
 */
template <typename T>
class DoubleBufferedHandle {
 public:
  DoubleBufferedHandle() = default;

  /**
   * @brief Publish new data.
   */
  void set(const T &value) {
    const uint8_t next_index = 1 - active_index_.load(std::memory_order_relaxed);
    buffers_[next_index] = value;
    active_index_.store(next_index, std::memory_order_release);
    valid_.store(true, std::memory_order_release);
  }

  /**
   * @brief Mark the handle as having no externally supplied data.
   */
  void clear() { valid_.store(false, std::memory_order_release); }

  /**
   * @brief Whether a valid data is currently available.
   */
  [[nodiscard]] bool hasValue() const { return valid_.load(std::memory_order_acquire); }

  /**
   * @brief Get the most recently published data.
   */
  [[nodiscard]] T get() const {
    const uint8_t active_index = active_index_.load(std::memory_order_acquire);
    return buffers_[active_index];
  }

 private:
  std::array<T, 2> buffers_{};
  std::atomic<uint8_t> active_index_{0};
  std::atomic<bool> valid_{false};
};

}  // namespace franky
