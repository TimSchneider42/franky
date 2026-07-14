#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "docstrings.hpp"
#include "franky.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the '_a' literal
using namespace franky;

void bind_motion_joint_pos(py::module &m) {
  py::class_<PositionWaypoint<JointState>>(m, "JointWaypoint", DOC(franky, PositionWaypoint))
      .def(
          py::init<>([](const JointState &target,
                        ReferenceType reference_type,
                        RelativeDynamicsFactor relative_dynamics_factor,
                        std::optional<franka::Duration>
                            minimum_time,
                        franka::Duration hold_target_duration,
                        std::optional<franka::Duration>
                            max_total_duration) {
            return PositionWaypoint<JointState>{
                {target, relative_dynamics_factor, minimum_time, hold_target_duration, max_total_duration},
                reference_type};
          }),
          "target"_a,
          py::arg_v("reference_type", ReferenceType::kAbsolute, "_franky.ReferenceType.Absolute"),
          "relative_dynamics_factor"_a = 1.0,
          "minimum_time"_a = std::nullopt,
          "hold_target_duration"_a = franka::Duration(0),
          "max_total_duration"_a = std::nullopt)
      .def_readonly("target", &PositionWaypoint<JointState>::target, DOC(franky, Waypoint, target))
      .def_readonly(
          "reference_type",
          &PositionWaypoint<JointState>::reference_type,
          DOC(franky, PositionWaypoint, reference_type))
      .def_readonly(
          "relative_dynamics_factor",
          &PositionWaypoint<JointState>::relative_dynamics_factor,
          DOC(franky, Waypoint, relative_dynamics_factor))
      .def_readonly("minimum_time", &PositionWaypoint<JointState>::minimum_time, DOC(franky, Waypoint, minimum_time))
      .def_readonly(
          "hold_target_duration",
          &PositionWaypoint<JointState>::hold_target_duration,
          DOC(franky, Waypoint, hold_target_duration))
      .def_readonly(
          "max_total_duration",
          &PositionWaypoint<JointState>::max_total_duration,
          DOC(franky, Waypoint, max_total_duration));

  py::class_<JointWaypointMotion, Motion<franka::JointPositions>, std::shared_ptr<JointWaypointMotion>>(
      m, "JointWaypointMotion", DOC(franky, JointWaypointMotion))
      .def(
          py::init<>([](const std::vector<PositionWaypoint<JointState>> &waypoints,
                        RelativeDynamicsFactor relative_dynamics_factor,
                        bool return_when_finished) {
            return std::make_shared<JointWaypointMotion>(waypoints, relative_dynamics_factor, return_when_finished);
          }),
          "waypoints"_a,
          "relative_dynamics_factor"_a = 1.0,
          "return_when_finished"_a = true,
          DOC(franky, JointWaypointMotion, JointWaypointMotion))
      .def_property_readonly("waypoints", &JointWaypointMotion::waypoints, DOC(franky, WaypointMotion, waypoints))
      .def_property_readonly(
          "relative_dynamics_factor",
          &JointWaypointMotion::relative_dynamics_factor,
          DOC(franky, PositionWaypointMotion, relative_dynamics_factor))
      .def_property_readonly(
          "return_when_finished",
          &JointWaypointMotion::return_when_finished,
          DOC(franky, WaypointMotion, return_when_finished));

  py::class_<JointMotion, JointWaypointMotion, std::shared_ptr<JointMotion>>(m, "JointMotion", DOC(franky, JointMotion))
      .def(
          py::init<const JointState &, ReferenceType, RelativeDynamicsFactor, bool>(),
          "target"_a,
          py::arg_v("reference_type", ReferenceType::kAbsolute, "_franky.ReferenceType.Absolute"),
          "relative_dynamics_factor"_a = 1.0,
          "return_when_finished"_a = true,
          DOC(franky, JointMotion, JointMotion))
      .def_property_readonly(
          "target",
          [](const JointMotion &motion) { return motion.waypoints().front().target; },
          "The target of this motion.")
      .def_property_readonly(
          "reference_type",
          [](const JointMotion &motion) { return motion.waypoints().front().reference_type; },
          "The reference type (absolute or relative) of this motion.");

  py::class_<
      StopMotion<franka::JointPositions>,
      Motion<franka::JointPositions>,
      std::shared_ptr<StopMotion<franka::JointPositions>>>(m, "JointStopMotion", DOC(franky, StopMotion_2))
      .def(py::init<RelativeDynamicsFactor>(), "relative_dynamics_factor"_a = 1.0, DOC(franky, StopMotion, StopMotion))
      .def_property_readonly(
          "relative_dynamics_factor",
          &StopMotion<franka::JointPositions>::relative_dynamics_factor,
          DOC(franky, PositionWaypointMotion, relative_dynamics_factor));
}
