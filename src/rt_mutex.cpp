#include "franky/rt_mutex.hpp"

#include <cstring>
#include <iostream>  // For std::cerr
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

#if !defined(_WIN32) && !defined(_WIN64)
// POSIX (Linux/macOS) specific includes for pthread
#include <pthread.h>
#endif

namespace franky {

// Common helper function to handle error checking for system calls
void checkRes(int res, const std::string &msg) {
  if (res != 0) {
    std::stringstream ss;
    ss << msg << ": " << std::strerror(res);
    throw std::runtime_error(ss.str());
  }
}

/**
 * @brief Patch std::mutex to allow for priority inheritance.
 *
 * Priority inheritance ensures that the thread currently holding this mutex
 * receives the same priority as the highest priority thread waiting for it.
 *
 * @param mutex Mutex to patch.
 */
void patchMutexRT(std::mutex &mutex) {
#if defined(_WIN32) || defined(_WIN64)
  // Static variable to ensure warning is printed only once
  static bool warningPrinted = false;

  if (!warningPrinted) {
    std::cerr << "Warning: Priority inheritance is not implemented on Windows. "
                 "Falling back on regular mutexes."
              << std::endl;
    warningPrinted = true;
  }

#else
  // POSIX (pthread) solution
  pthread_mutexattr_t attr;

  checkRes(pthread_mutex_destroy(mutex.native_handle()), "Failed to destroy native mutex.");
  checkRes(pthread_mutexattr_init(&attr), "Failed to initialize pthread mutex attributes");
  checkRes(
      pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT),
      "Failed to enable priority inheritance for pthread mutex");
  checkRes(pthread_mutex_init(mutex.native_handle(), &attr), "Failed to initialize mutex");

  pthread_mutexattr_destroy(&attr);
#endif
}

}  // namespace franky
