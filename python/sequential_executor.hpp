#pragma once

#include <atomic>
#include <functional>
#include <thread>

#if !defined(_WIN32) && !defined(_WIN64)
#include <semaphore.h>
#else
#include <condition_variable>
#include <mutex>
#endif

#include "concurrent_queue.hpp"
#include "rt_function_queue.hpp"

/**
 * @brief Executes callbacks sequentially on a dedicated thread, accepting them from real-time
 * threads.
 *
 * Callbacks enter through a bounded lock-free queue (add() is real-time safe), get moved by a
 * relay thread into an unbounded queue (absorbing arbitrarily slow consumers, e.g. Python
 * callbacks serialized behind the GIL), and are finally run one by one on the executor thread.
 */
class SequentialExecutor {
 public:
  SequentialExecutor();

  ~SequentialExecutor();

  /**
   * @brief Join both worker threads. Safe to call repeatedly and concurrently.
   *
   * Does not fence out producers: add() stays callable and cheap afterwards, it just stops having a
   * consumer. Callbacks still queued or already relayed are not run.
   */
  void shutdown();

  /** Destroy queued callbacks after shutdown. The caller must hold the Python GIL. */
  void clearPending();

  /**
   * @brief Enqueue a callback for execution. Real-time safe: lock-free, no allocation, no page
   * faults. The callback and its captured state must fit kSlotSize bytes (checked at compile
   * time) and must not capture anything whose copy or destructor blocks (e.g. py::object).
   *
   * @return false if the input queue is full; the callback is dropped and counted.
   */
  template <typename F>
  bool add(F &&function) {
    // Safe to call at any time, including concurrently with and after shutdown(): late call merely enqueues a callback
    // that no one will run.
    const bool added = input_queue_.tryEmplace(std::forward<F>(function));
    if (added) {
      notifyRelay();
    } else {
      dropped_.fetch_add(1, std::memory_order_relaxed);
    }
    return added;
  }

  //! Total number of callbacks dropped because the input queue was full.
  [[nodiscard]] size_t droppedCount() const { return dropped_.load(std::memory_order_relaxed); }

 private:
  static constexpr size_t kSlotSize = 8192;
  static constexpr size_t kCapacity = 256;

  RTFunctionQueue<kSlotSize, kCapacity> input_queue_;
  ConcurrentQueue<std::function<void()>> queue_;
  std::atomic<bool> terminate_{false};
  std::atomic<bool> shutdown_started_{false};
  std::atomic<bool> shutdown_complete_{false};
  std::atomic<size_t> dropped_{0};
  std::thread execute_thread_;
  std::thread relay_thread_;

#if !defined(_WIN32) && !defined(_WIN64)
  sem_t relay_semaphore_;
#else
  std::mutex relay_mutex_;
  std::condition_variable relay_condition_;
  size_t relay_notifications_{0};
#endif

  void notifyRelay();
  void waitForRelay();
  void relay();
  void execute();
};
