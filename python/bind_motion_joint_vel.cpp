#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "docstrings.hpp"
#include "franky.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the '_a' literal
using namespace franky;

void bind_motion_joint_vel(py::module &m) {
  py::class_<VelocityWaypoint<Vector7d>>(m, "JointVelocityWaypoint", DOC(franky, Waypoint))
      .def(
          py::init<>([](const Vector7d &target,
                        RelativeDynamicsFactor relative_dynamics_factor,
                        std::optional<franka::Duration>
                            minimum_time,
                        franka::Duration hold_target_duration,
                        std::optional<franka::Duration>
                            max_total_duration) {
            return VelocityWaypoint<Vector7d>{
                target, relative_dynamics_factor, minimum_time, hold_target_duration, max_total_duration};
          }),
          "target"_a,
          "relative_dynamics_factor"_a = 1.0,
          "minimum_time"_a = std::nullopt,
          "hold_target_duration"_a = franka::Duration(0),
          "max_total_duration"_a = std::nullopt)
      .def_readonly("target", &VelocityWaypoint<Vector7d>::target, DOC(franky, Waypoint, target))
      .def_readonly(
          "relative_dynamics_factor",
          &VelocityWaypoint<Vector7d>::relative_dynamics_factor,
          DOC(franky, Waypoint, relative_dynamics_factor))
      .def_readonly("minimum_time", &VelocityWaypoint<Vector7d>::minimum_time, DOC(franky, Waypoint, minimum_time))
      .def_readonly(
          "hold_target_duration",
          &VelocityWaypoint<Vector7d>::hold_target_duration,
          DOC(franky, Waypoint, hold_target_duration))
      .def_readonly(
          "max_total_duration",
          &VelocityWaypoint<Vector7d>::max_total_duration,
          DOC(franky, Waypoint, max_total_duration));

  py::class_<
      JointVelocityWaypointMotion,
      Motion<franka::JointVelocities>,
      std::shared_ptr<JointVelocityWaypointMotion>>(
      m, "JointVelocityWaypointMotion", DOC(franky, JointVelocityWaypointMotion))
      .def(
          py::init<>([](const std::vector<VelocityWaypoint<Vector7d>> &waypoints,
                        RelativeDynamicsFactor relative_dynamics_factor) {
            return std::make_shared<JointVelocityWaypointMotion>(waypoints, relative_dynamics_factor);
          }),
          "waypoints"_a,
          "relative_dynamics_factor"_a = 1.0,
          DOC(franky, JointVelocityWaypointMotion, JointVelocityWaypointMotion))
      .def_property_readonly(
          "waypoints", &JointVelocityWaypointMotion::waypoints, DOC(franky, WaypointMotion, waypoints))
      .def_property_readonly(
          "relative_dynamics_factor",
          &JointVelocityWaypointMotion::relative_dynamics_factor,
          DOC(franky, VelocityWaypointMotion, relative_dynamics_factor));

  py::class_<JointVelocityMotion, JointVelocityWaypointMotion, std::shared_ptr<JointVelocityMotion>>(
      m, "JointVelocityMotion", DOC(franky, JointVelocityMotion))
      .def(
          py::init<const Vector7d &, franka::Duration, RelativeDynamicsFactor>(),
          "target"_a,
          "duration"_a = franka::Duration(1000),
          "relative_dynamics_factor"_a = 1.0,
          DOC(franky, JointVelocityMotion, JointVelocityMotion))
      .def_property_readonly(
          "target",
          [](const JointVelocityMotion &motion) { return motion.waypoints().front().target; },
          "The target of this motion.")
      .def_property_readonly(
          "duration",
          [](const JointVelocityMotion &motion) { return motion.waypoints().front().hold_target_duration; },
          "For how long the target is held after it has been reached.");

  py::class_<
      StopMotion<franka::JointVelocities>,
      Motion<franka::JointVelocities>,
      std::shared_ptr<StopMotion<franka::JointVelocities>>>(m, "JointVelocityStopMotion", DOC(franky, StopMotion_3))
      .def(
          py::init<RelativeDynamicsFactor>(), "relative_dynamics_factor"_a = 1.0, DOC(franky, StopMotion, StopMotion_2))
      .def_property_readonly(
          "relative_dynamics_factor",
          &StopMotion<franka::JointVelocities>::relative_dynamics_factor,
          DOC(franky, VelocityWaypointMotion, relative_dynamics_factor));
}
