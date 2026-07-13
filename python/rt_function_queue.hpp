#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <new>
#include <type_traits>
#include <utility>

#if !defined(_WIN32) && !defined(_WIN64)
#include <sys/mman.h>
#endif

/**
 * @brief Bounded, lock-free, multi-producer multi-consumer queue of type-erased callables.
 *
 * Designed for real-time producers: tryEmplace() never blocks, never allocates, and never touches
 * unmapped memory. The callable (including all of its captured state) is constructed in-place in a
 * fixed-size slot of a preallocated ring buffer. The buffer is pre-faulted and locked into RAM
 * (best-effort) at construction, so pushing cannot cause page faults.
 *
 * Closures are moved out into a std::function on the consumer side, where allocation is
 * acceptable.
 *
 * Based on Dmitry Vyukov's bounded MPMC queue: each slot carries a sequence number that producers
 * and consumers use to claim and publish slots without locks.
 *
 * @tparam SlotSize Storage bytes available per slot for the closure. Exceeding it is a compile
 *                  error at the tryEmplace() call site.
 * @tparam Capacity Number of slots. Must be a power of two.
 */
template <size_t SlotSize, size_t Capacity>
class RTFunctionQueue {
  static_assert(Capacity >= 2 && (Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");

 public:
  RTFunctionQueue() {
    for (size_t i = 0; i < Capacity; ++i) slots_[i].seq.store(i, std::memory_order_relaxed);
    // Fault in every page of the ring now so producers never hit a soft page fault. mlock
    // additionally prevents the pages from being swapped out; it may fail without
    // CAP_IPC_LOCK / sufficient RLIMIT_MEMLOCK, in which case pre-faulting is all we get.
    auto *base = reinterpret_cast<volatile std::byte *>(slots_.data());
    for (size_t i = 0; i < sizeof(slots_); i += kPageSize) base[i] = base[i];
#if !defined(_WIN32) && !defined(_WIN64)
    locked_ = mlock(slots_.data(), sizeof(slots_)) == 0;
#endif
  }

  ~RTFunctionQueue() {
    std::function<void()> discard;
    while (tryPop(discard));
#if !defined(_WIN32) && !defined(_WIN64)
    if (locked_) munlock(slots_.data(), sizeof(slots_));
#endif
  }

  RTFunctionQueue(const RTFunctionQueue &) = delete;
  RTFunctionQueue &operator=(const RTFunctionQueue &) = delete;

  /**
   * @brief Construct a callable in-place in the queue. Real-time safe: lock-free, no allocation.
   *
   * @return false if the queue is full (the callable is not enqueued).
   */
  template <typename F>
  bool tryEmplace(F &&function) {
    using Fn = std::decay_t<F>;
    static_assert(sizeof(Fn) <= SlotSize, "Closure exceeds slot size; increase SlotSize");
    static_assert(alignof(Fn) <= alignof(std::max_align_t), "Closure is over-aligned for slot storage");
    // A throwing move here would leave a claimed slot unpublished and wedge the queue.
    static_assert(std::is_nothrow_move_constructible_v<Fn>, "Closure must be nothrow move constructible");

    size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
    for (;;) {
      Slot &slot = slots_[pos & (Capacity - 1)];
      size_t seq = slot.seq.load(std::memory_order_acquire);
      auto diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
      if (diff == 0) {
        if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
          new (slot.storage) Fn(std::forward<F>(function));
          slot.extract = [](void *src, std::function<void()> &dst) {
            auto &fn = *static_cast<Fn *>(src);
            dst = std::move(fn);
            fn.~Fn();
          };
          slot.seq.store(pos + 1, std::memory_order_release);
          return true;
        }
      } else if (diff < 0) {
        return false;
      } else {
        pos = enqueue_pos_.load(std::memory_order_relaxed);
      }
    }
  }

  /**
   * @brief Move the oldest callable out of the queue into @p out. Not real-time safe: assigning to
   * the std::function allocates.
   *
   * @return false if the queue is empty.
   */
  bool tryPop(std::function<void()> &out) {
    size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
    for (;;) {
      Slot &slot = slots_[pos & (Capacity - 1)];
      size_t seq = slot.seq.load(std::memory_order_acquire);
      auto diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
      if (diff == 0) {
        if (dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
          slot.extract(slot.storage, out);
          slot.seq.store(pos + Capacity, std::memory_order_release);
          return true;
        }
      } else if (diff < 0) {
        return false;
      } else {
        pos = dequeue_pos_.load(std::memory_order_relaxed);
      }
    }
  }

 private:
  static constexpr size_t kPageSize = 4096;
  static constexpr size_t kCacheLineSize = 64;

  struct Slot {
    std::atomic<size_t> seq;
    void (*extract)(void *src, std::function<void()> &dst);
    alignas(std::max_align_t) std::byte storage[SlotSize];
  };

  std::array<Slot, Capacity> slots_;
  alignas(kCacheLineSize) std::atomic<size_t> enqueue_pos_{0};
  alignas(kCacheLineSize) std::atomic<size_t> dequeue_pos_{0};
  bool locked_{false};
};
