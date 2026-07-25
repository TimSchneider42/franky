#pragma once

#include <condition_variable>
#include <optional>
#include <queue>

#include "franky.hpp"

template <typename T>
class ConcurrentQueue {
 public:
  ConcurrentQueue() { franky::patchMutexRT(mutex_); }

  void push(T item) {
    std::lock_guard lock(mutex_);
    queue_.push(std::move(item));
    condition_.notify_one();
  }

  std::optional<T> pop() {
    std::unique_lock lock(mutex_);
    condition_.wait(lock, [this]() { return closed_ || !queue_.empty(); });
    if (closed_) return std::nullopt;
    T item = std::move(queue_.front());
    queue_.pop();
    return item;
  }

  template <class Rep, class Period>
  std::optional<T> pop(const std::chrono::duration<Rep, Period> &timeout) {
    std::unique_lock lock(mutex_);
    if (condition_.wait_for(lock, timeout, [this]() { return !queue_.empty(); })) {
      T item = std::move(queue_.front());
      queue_.pop();
      return item;
    }
    return std::nullopt;
  }

  void clear() {
    std::queue<T> empty;
    std::lock_guard lock(mutex_);
    queue_.swap(empty);
  }

  void close() {
    {
      std::lock_guard lock(mutex_);
      closed_ = true;
    }
    condition_.notify_all();
  }

 private:
  std::queue<T> queue_;
  std::mutex mutex_;
  std::condition_variable condition_;
  bool closed_{false};
};
