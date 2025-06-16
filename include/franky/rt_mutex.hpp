#pragma once

#include <mutex>
#include <string>

namespace franky {

/**
 * Common helper function to handle error checking for system calls.
 *
 * @param res Result of the call.
 * @param msg Message to print for error.
 */
void checkRes(int res, const std::string &msg);

/**
 * @brief Patch std::mutex to allow for priority inheritance.
 *
 * Priority inheritance ensures that the thread currently holding this mutex
 * receives the same priority as the highest priority thread waiting for it.
 *
 * @param mutex Mutex to patch.
 */
void patchMutexRT(std::mutex &mutex);
}  // namespace franky
