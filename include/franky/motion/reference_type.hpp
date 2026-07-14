#pragma once

namespace franky {

/**
 * @brief Enum class for reference types.
 *
 * This enum class defines the reference types for motions (absolute or
 * relative).
 */
enum class ReferenceType {
  /** The target is given in the robot's base frame (or as absolute joint positions). */
  kAbsolute,
  /** The target is given relative to the robot's state when the motion starts. */
  kRelative
};

}  // namespace franky
