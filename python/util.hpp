#pragma once

#include <pybind11/pybind11.h>

#include <algorithm>
#include <chrono>
#include <future>
#include <iostream>
#include <optional>

using namespace franky;

inline void checkPythonSignals() {
  if (!Py_IsInitialized()) return;
  pybind11::gil_scoped_acquire gil_acquire;
  if (PyErr_CheckSignals() == -1) throw pybind11::error_already_set();
}

template <typename WaitFor>
bool interruptibleWait(WaitFor &&wait_for, const std::optional<double> &timeout = std::nullopt) {
  using Clock = std::chrono::steady_clock;
  constexpr auto poll_interval = std::chrono::milliseconds(100);

  std::optional<Clock::time_point> deadline;
  if (timeout.has_value()) {
    if (timeout.value() <= 0.0) return wait_for(std::chrono::duration<double>::zero());
    deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(*timeout));
  }

  while (true) {
    auto wait_duration = std::chrono::duration_cast<Clock::duration>(poll_interval);
    if (deadline.has_value()) {
      const auto remaining = deadline.value() - Clock::now();
      if (remaining <= Clock::duration::zero()) return wait_for(std::chrono::duration<double>::zero());
      wait_duration = std::min(wait_duration, remaining);
    }

    if (wait_for(wait_duration)) return true;
    checkPythonSignals();
  }
}

template <typename Future>
bool waitForFutureInterruptibly(const Future &future, const std::optional<double> &timeout = std::nullopt) {
  return interruptibleWait(
      [&future](const auto &wait_duration) { return future.wait_for(wait_duration) != std::future_status::timeout; },
      timeout);
}

template <typename T>
std::string strFromStream(const T &obj) {
  std::stringstream ss;
  ss << obj;
  return ss.str();
}
