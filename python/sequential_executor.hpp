#pragma once

#include <functional>
#include <thread>

#include "concurrent_queue.hpp"

class SequentialExecutor {
 public:
  SequentialExecutor();

  ~SequentialExecutor();

  void add(const std::function<void()> &function);

 private:
  ConcurrentQueue<std::function<void()>> queue_;
  bool terminate_{false};
  std::thread thread_;

  void execute();
};
