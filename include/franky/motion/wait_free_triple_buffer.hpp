#pragma once

#include <array>
#include <atomic>
#include <cstdint>

namespace franky {

/**
 * @brief Wait-free, Single-Producer Single-Consumer (SPSC) triple buffer.
 *
 * This buffer allows a single user thread to safely write data without ever blocking,
 * and a single real-time loop to safely read the latest available data without locking
 * or experiencing torn reads.
 */
template <typename T>
class WaitFreeTripleBuffer {
 public:
  WaitFreeTripleBuffer() = default;

  /**
   * @brief Publish new data.
   */
  void set(const T &value) {
    // 1. Write the payload into our privately owned write buffer
    buffers_[active_write_index_] = value;

    // 2. Pack the write index (bits 0-1) and set the "new data" flag (bit 2)
    const uint8_t new_state = active_write_index_ | 0x04;

    // 3. Atomically publish our buffer to the shared pool and take the old buffer.
    // Memory order acq_rel ensures the payload write finishes before the index swap.
    const uint8_t old_state = shared_state_.exchange(new_state, std::memory_order_acq_rel);

    // 4. Our next write will go to the buffer that the reader just gave up (or never used).
    active_write_index_ = old_state & 0x03;

    valid_.store(true, std::memory_order_release);
  }

  /**
   * @brief Mark the handle as having no externally supplied data.
   */
  void clear() { valid_.store(false, std::memory_order_release); }

  /**
   * @brief Whether valid data is currently available.
   */
  [[nodiscard]] bool hasValue() const { return valid_.load(std::memory_order_acquire); }

  /**
   * @brief Get the most recently published data.
   * * Note: This method mutates internal read indices to securely take ownership
   * of the newest buffer, hence it cannot be marked const.
   */
  [[nodiscard]] T get() {
    const uint8_t current_state = shared_state_.load(std::memory_order_acquire);

    // If the writer has published a new buffer since we last checked...
    if ((current_state & 0x04) != 0) {
      // Atomically hand our current read buffer back to the shared pool
      // (with the "new" flag cleared) and take ownership of the newest buffer.
      const uint8_t old_state = shared_state_.exchange(active_read_index_, std::memory_order_acq_rel);
      active_read_index_ = old_state & 0x03;
    }

    return buffers_[active_read_index_];
  }

 private:
  std::array<T, 3> buffers_{};

  // Start indices offset from one another so they never point to the same buffer initially.
  uint8_t active_write_index_{1};
  uint8_t active_read_index_{2};

  // Packs both the shared clean index (bits 0-1) and the "new data" flag (bit 2).
  // Initial state: index 0, new data flag = 0.
  std::atomic<uint8_t> shared_state_{0};

  std::atomic<bool> valid_{false};
};

}  // namespace franky
