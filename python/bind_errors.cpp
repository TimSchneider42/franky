#include <franka/errors.h>
#include <franka/exception.h>
#include <pybind11/pybind11.h>

#include "franky.hpp"
#include "macros.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the '_a' literal
using namespace franky;

#define ERRORS_0_8                                                                                                    \
  joint_position_limits_violation, cartesian_position_limits_violation, self_collision_avoidance_violation,           \
      joint_velocity_violation, cartesian_velocity_violation, force_control_safety_violation, joint_reflex,           \
      cartesian_reflex, max_goal_pose_deviation_violation, max_path_pose_deviation_violation,                         \
      cartesian_velocity_profile_safety_violation, joint_position_motion_generator_start_pose_invalid,                \
      joint_motion_generator_position_limits_violation, joint_motion_generator_velocity_limits_violation,             \
      joint_motion_generator_velocity_discontinuity, joint_motion_generator_acceleration_discontinuity,               \
      cartesian_position_motion_generator_start_pose_invalid, cartesian_motion_generator_elbow_limit_violation,       \
      cartesian_motion_generator_velocity_limits_violation, cartesian_motion_generator_velocity_discontinuity,        \
      cartesian_motion_generator_acceleration_discontinuity, cartesian_motion_generator_elbow_sign_inconsistent,      \
      cartesian_motion_generator_start_elbow_invalid, cartesian_motion_generator_joint_position_limits_violation,     \
      cartesian_motion_generator_joint_velocity_limits_violation,                                                     \
      cartesian_motion_generator_joint_velocity_discontinuity,                                                        \
      cartesian_motion_generator_joint_acceleration_discontinuity, cartesian_position_motion_generator_invalid_frame, \
      force_controller_desired_force_tolerance_violation, controller_torque_discontinuity,                            \
      start_elbow_sign_inconsistent, communication_constraints_violation, power_limit_violation,                      \
      joint_p2p_insufficient_torque_for_planning, tau_j_range_violation, instability_detected,                        \
      joint_move_in_wrong_direction

#ifdef FRANKA_0_9
#define ERRORS                                                                                                        \
  ERRORS_0_8, cartesian_spline_motion_generator_violation, joint_via_motion_generator_planning_joint_limit_violation, \
      base_acceleration_initialization_timeout, base_acceleration_invalid_reading
#define NUM_ERRORS 41
#else
#define ERRORS ERRORS_0_8
#define NUM_ERRORS 37
#endif

#define ADD_ERROR(unused, name) errors.def_property_readonly(#name, [](const franka::Errors &e) { return e.name; });
#define UNPACK_ERRORS_INNER(tuple, itr, name) , tuple[itr + 1].cast<bool>()
#define UNPACK_ERRORS_1(tuple, name0, ...)                                                                 \
  franka::Errors {                                                                                         \
    std::array<bool, NUM_ERRORS> { tuple[0].cast<bool>() MAP_C1(UNPACK_ERRORS_INNER, tuple, __VA_ARGS__) } \
  }
#define UNPACK_ERRORS(tuple, ...) UNPACK_ERRORS_1(tuple, __VA_ARGS__)

void bind_errors(py::module &m) {
  py::class_<franka::Errors> errors(m, "Errors");
  errors.def(py::init<>());
  MAP(ADD_ERROR, ERRORS)
  errors.def("__repr__", [](const franka::Errors &errors) { return std::string(errors); });
  errors.def(
      py::pickle(
          [](const franka::Errors &errors) {  // __getstate__
            return PACK_TUPLE(errors, ERRORS);
          },
          [](const py::tuple &t) {  // __setstate__
            if (t.size() != COUNT(ERRORS)) throw std::runtime_error("Invalid state!");
            return UNPACK_ERRORS(t, ERRORS);
          }));

  py::register_exception<franka::Exception>(m, "Exception");
  py::register_exception<franka::CommandException>(m, "CommandException");
  py::register_exception<franka::ControlException>(m, "ControlException");
  py::register_exception<franka::IncompatibleVersionException>(m, "IncompatibleVersionException");
  py::register_exception<franka::InvalidOperationException>(m, "InvalidOperationException");
  py::register_exception<franka::ModelException>(m, "ModelException");
  py::register_exception<franka::NetworkException>(m, "NetworkException");
  py::register_exception<franka::ProtocolException>(m, "ProtocolException");
  py::register_exception<franka::RealtimeException>(m, "RealtimeException");
  py::register_exception<InvalidMotionTypeException>(m, "InvalidMotionTypeException");
  py::register_exception<ReactionRecursionException>(m, "ReactionRecursionException");
  py::register_exception<GripperException>(m, "GripperException");
}
