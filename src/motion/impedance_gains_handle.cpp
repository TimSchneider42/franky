#include "franky/motion/impedance_gains_handle.hpp"

namespace franky {

// --- CartesianImpedanceGainsHandle ---

void CartesianImpedanceGainsHandle::set(const CartesianImpedanceGains &gains) {
  const uint8_t next_index = 1 - active_index_.load(std::memory_order_relaxed);
  buffers_[next_index] = gains;
  active_index_.store(next_index, std::memory_order_release);
  valid_.store(true, std::memory_order_release);
}

void CartesianImpedanceGainsHandle::clear() { valid_.store(false, std::memory_order_release); }

bool CartesianImpedanceGainsHandle::hasGains() const { return valid_.load(std::memory_order_acquire); }

CartesianImpedanceGains CartesianImpedanceGainsHandle::get() const {
  const uint8_t active_index = active_index_.load(std::memory_order_acquire);
  return buffers_[active_index];
}

// --- JointImpedanceGainsHandle ---

void JointImpedanceGainsHandle::set(const JointImpedanceGains &gains) {
  const uint8_t next_index = 1 - active_index_.load(std::memory_order_relaxed);
  buffers_[next_index] = gains;
  active_index_.store(next_index, std::memory_order_release);
  valid_.store(true, std::memory_order_release);
}

void JointImpedanceGainsHandle::clear() { valid_.store(false, std::memory_order_release); }

bool JointImpedanceGainsHandle::hasGains() const { return valid_.load(std::memory_order_acquire); }

JointImpedanceGains JointImpedanceGainsHandle::get() const {
  const uint8_t active_index = active_index_.load(std::memory_order_acquire);
  return buffers_[active_index];
}

}  // namespace franky
