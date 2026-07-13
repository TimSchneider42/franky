#include "sequential_executor.hpp"

#include <chrono>
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

SequentialExecutor::SequentialExecutor()
    : execute_thread_(&SequentialExecutor::execute, this), relay_thread_(&SequentialExecutor::relay, this) {}

SequentialExecutor::~SequentialExecutor() {
  terminate_.store(true, std::memory_order_release);
  relay_thread_.join();
  execute_thread_.join();
}

void SequentialExecutor::relay() {
  tryMakeThreadRT();
  size_t reported_dropped = 0;
  std::function<void()> function;
  while (!terminate_.load(std::memory_order_acquire)) {
    bool idle = true;
    while (input_queue_.tryPop(function)) {
      queue_.push(std::move(function));
      idle = false;
    }
    auto dropped = dropped_.load(std::memory_order_relaxed);
    if (dropped > reported_dropped) {
      std::cerr << "franky: callback input queue overflowed, " << dropped
                << " callback invocation(s) dropped so far. The callback relay thread is not getting enough CPU time."
                << std::endl;
      reported_dropped = dropped;
    }
    if (idle) std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
}

void SequentialExecutor::execute() {
  while (!terminate_.load(std::memory_order_acquire)) {
    auto callback = queue_.pop(std::chrono::microseconds(100));
    if (callback.has_value()) (*callback)();
  }
}
