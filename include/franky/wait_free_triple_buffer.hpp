#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>

namespace franky {

/**
 * @brief Triple buffer connecting concurrent user threads with a single real-time thread.
 *
 * The core exchange is a wait-free, Single-Producer Single-Consumer (SPSC) triple buffer: one
 * writer can publish data without ever blocking, and one reader can obtain the latest available
 * data without locking or experiencing torn reads. On top of that, a write mutex and a read mutex
 * serialize concurrent writers and concurrent readers, respectively, so that multiple user threads
 * may safely share either side.
 *
 * The real-time thread must not take these mutexes, as it could then be blocked by user threads.
 * Instead, it must exclusively use the unsafe variants (setUnsafe(), getUnsafe(),
 * getLastWrittenUnsafe()), which is safe as long as it is the only thread on its side of the
 * buffer: all other threads on the same side must use the locking variants, and no other thread
 * may use the unsafe variants concurrently.
 */
template <typename T>
class WaitFreeTripleBuffer {
 public:
  WaitFreeTripleBuffer() = default;

  explicit WaitFreeTripleBuffer(const T &initial_value) { setUnsafe(initial_value); }

  /**
   * @brief Publish new data.
   *
   * Takes the write mutex to serialize concurrent writers. Must not be called from the real-time
   * thread; use setUnsafe() there instead.
   */
  void set(const T &value) {
    std::lock_guard lock(write_mutex_);
    setUnsafe(value);
  }

  /**
   * @brief Publish new data without taking the write mutex.
   *
   * Must only be called if no other thread is writing concurrently. This is the variant the
   * real-time thread has to use, as it must not block on the write mutex.
   */
  void setUnsafe(const T &value) {
    // 1. Write the payload into our privately owned write buffer
    buffers_[active_write_index_] = value;
    last_written_index_ = active_write_index_;

    // 2. Pack the write index (bits 0-1) and set the "new data" flag (bit 2)
    const uint8_t new_state = active_write_index_ | 0x04;

    // 3. Atomically publish our buffer to the shared pool and take the old buffer.
    // Memory order acq_rel ensures the payload write finishes before the index swap.
    const uint8_t old_state = shared_state_.exchange(new_state, std::memory_order_acq_rel);

    // 4. Our next write will go to the buffer that the reader just gave up (or never used).
    active_write_index_ = old_state & 0x03;
  }

  /**
   * @brief Get the most recently written value from the writer's side.
   *
   * Takes the write mutex, as it accesses writer-owned state. Unlike get(), this does not consume
   * the "new data" flag or interact with the reader in any way. Before the first set(), this
   * returns a default-constructed T. Must not be called from the real-time thread; use
   * getLastWrittenUnsafe() there instead.
   */
  [[nodiscard]] T getLastWritten() const {
    std::lock_guard lock(write_mutex_);
    return getLastWrittenUnsafe();
  }

  /**
   * @brief Get the most recently written value from the writer's side without taking the write
   * mutex.
   *
   * Must only be called if no other thread is writing concurrently. This is the variant the
   * real-time thread has to use, as it must not block on the write mutex.
   */
  [[nodiscard]] T getLastWrittenUnsafe() const { return buffers_[last_written_index_]; }

  /**
   * @brief Get the most recently published data.
   *
   * Takes the read mutex to serialize concurrent readers. Must not be called from the real-time
   * thread; use getUnsafe() there instead.
   *
   * Note: This method mutates internal read indices to securely take ownership of the newest
   * buffer, hence it cannot be marked const.
   */
  [[nodiscard]] T get() {
    std::lock_guard lock(read_mutex_);
    return getUnsafe();
  }

  /**
   * @brief Get the most recently published data without taking the read mutex.
   *
   * Must only be called if no other thread is reading concurrently. This is the variant the
   * real-time thread has to use, as it must not block on the read mutex.
   */
  [[nodiscard]] T getUnsafe() {
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

  // Owned by the writer side: index of the buffer holding the last written value.
  // A published buffer is never written again until the writer reclaims it in a
  // later set(), so the writer may safely read it back without synchronization.
  uint8_t last_written_index_{0};

  // Packs both the shared clean index (bits 0-1) and the "new data" flag (bit 2).
  // Initial state: index 0, new data flag = 0.
  std::atomic<uint8_t> shared_state_{0};

  // Serialize concurrent writers and readers, respectively. The real-time thread must not take
  // either mutex and uses the unsafe operations instead.
  mutable std::mutex write_mutex_;
  std::mutex read_mutex_;
};

}  // namespace franky
