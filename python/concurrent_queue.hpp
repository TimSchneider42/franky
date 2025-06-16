#pragma once

#include <condition_variable>
#include <optional>
#include <queue>

#include "franky.hpp"

template <typename T>
class ConcurrentQueue {
 public:
  ConcurrentQueue() { franky::patchMutexRT(mutex_); }

  void push(const T &item) {
    std::lock_guard lock(mutex_);
    queue_.push(item);
    condition_.notify_one();
  }

  T pop() {
    std::unique_lock lock(mutex_);
    condition_.wait(lock, [this]() { return !queue_.empty(); });
    T item = queue_.front();
    queue_.pop();
    return item;
  }

  template <class Rep, class Period>
  std::optional<T> pop(const std::chrono::duration<Rep, Period> &timeout) {
    std::unique_lock lock(mutex_);
    if (condition_.wait_for(lock, timeout, [this]() { return !queue_.empty(); })) {
      T item = queue_.front();
      queue_.pop();
      return item;
    }
    return std::nullopt;
  }

 private:
  std::queue<T> queue_;
  std::mutex mutex_;
  std::condition_variable condition_;
};
