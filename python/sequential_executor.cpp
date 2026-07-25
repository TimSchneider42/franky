#include "sequential_executor.hpp"

#include <cerrno>
#include <iostream>

#if !defined(_WIN32) && !defined(_WIN64)
#include <pthread.h>
#endif

namespace {

// Best-effort: elevate the current thread so the relay is not starved by regular threads while
// the real-time producers fill the bounded input queue. Requires rtprio privileges (which
// libfranka needs anyway); silently keeps the default policy if unavailable.
void tryMakeThreadRT() {
#if !defined(_WIN32) && !defined(_WIN64)
  sched_param param{};
  param.sched_priority = sched_get_priority_max(SCHED_FIFO) - 1;
  if (param.sched_priority > 0) pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
#endif
}

}  // namespace

SequentialExecutor::SequentialExecutor() {
#if !defined(_WIN32) && !defined(_WIN64)
  sem_init(&relay_semaphore_, 0, 0);
#endif
  execute_thread_ = std::thread(&SequentialExecutor::execute, this);
  relay_thread_ = std::thread(&SequentialExecutor::relay, this);
}

SequentialExecutor::~SequentialExecutor() {
  shutdown();
  clearPending();
}

void SequentialExecutor::shutdown() {
  bool expected = false;
  if (!shutdown_started_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
    while (!shutdown_complete_.load(std::memory_order_acquire)) std::this_thread::yield();
    return;
  }

  terminate_.store(true, std::memory_order_release);
  notifyRelay();
  if (relay_thread_.joinable()) relay_thread_.join();
  queue_.close();
  if (execute_thread_.joinable()) execute_thread_.join();
  // No sem_destroy: the semaphore must stay valid for late sem_post() calls from producers. This
  // object is leaked anyway, so there is nothing to reclaim.
  shutdown_complete_.store(true, std::memory_order_release);
}

void SequentialExecutor::clearPending() {
  std::function<void()> pending;
  while (input_queue_.tryPop(pending)) pending = {};
  queue_.clear();
}

void SequentialExecutor::notifyRelay() {
#if !defined(_WIN32) && !defined(_WIN64)
  sem_post(&relay_semaphore_);
#else
  {
    std::lock_guard lock(relay_mutex_);
    ++relay_notifications_;
  }
  relay_condition_.notify_one();
#endif
}

void SequentialExecutor::waitForRelay() {
#if !defined(_WIN32) && !defined(_WIN64)
  while (sem_wait(&relay_semaphore_) != 0 && errno == EINTR) {
  }
#else
  std::unique_lock lock(relay_mutex_);
  relay_condition_.wait(lock, [this]() { return relay_notifications_ != 0; });
  --relay_notifications_;
#endif
}

void SequentialExecutor::relay() {
  tryMakeThreadRT();
  size_t reported_dropped = 0;
  std::function<void()> function;
  while (true) {
    waitForRelay();
    if (terminate_.load(std::memory_order_acquire)) break;
    // Actually allocated memory for the callback that the user registered
    while (input_queue_.tryPop(function)) {
      queue_.push(std::move(function));
    }
    auto dropped = dropped_.load(std::memory_order_relaxed);
    if (dropped > reported_dropped) {
      std::cerr << "franky: callback input queue overflowed, " << dropped
                << " callback invocation(s) dropped so far. The callback relay thread is not getting enough CPU time."
                << std::endl;
      reported_dropped = dropped;
    }
  }
}

void SequentialExecutor::execute() {
  while (auto callback = queue_.pop()) (*callback)();
}
