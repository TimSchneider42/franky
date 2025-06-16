#include "sequential_executor.hpp"

SequentialExecutor::SequentialExecutor() : thread_(&SequentialExecutor::execute, this) {}

SequentialExecutor::~SequentialExecutor() {
  terminate_ = true;
  thread_.join();
}

void SequentialExecutor::execute() {
  while (!terminate_) {
    auto callback = queue_.pop(std::chrono::microseconds(100));
    if (callback.has_value()) (*callback)();
  }
}

void SequentialExecutor::add(const std::function<void()> &function) { queue_.push(function); }
