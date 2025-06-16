#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "franky.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the '_a' literal
using namespace franky;

void bind_motion_joint_vel(py::module &m) {
  py::class_<VelocityWaypoint<Vector7d>>(m, "JointVelocityWaypoint")
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
      .def_readonly("target", &VelocityWaypoint<Vector7d>::target)
      .def_readonly("relative_dynamics_factor", &VelocityWaypoint<Vector7d>::relative_dynamics_factor)
      .def_readonly("minimum_time", &VelocityWaypoint<Vector7d>::minimum_time)
      .def_readonly("hold_target_duration", &VelocityWaypoint<Vector7d>::hold_target_duration)
      .def_readonly("max_total_duration", &VelocityWaypoint<Vector7d>::max_total_duration);

  py::class_<
      JointVelocityWaypointMotion,
      Motion<franka::JointVelocities>,
      std::shared_ptr<JointVelocityWaypointMotion>>(m, "JointVelocityWaypointMotion")
      .def(
          py::init<>([](const std::vector<VelocityWaypoint<Vector7d>> &waypoints,
                        RelativeDynamicsFactor relative_dynamics_factor) {
            return std::make_shared<JointVelocityWaypointMotion>(waypoints, relative_dynamics_factor);
          }),
          "waypoints"_a,
          "relative_dynamics_factor"_a = 1.0);

  py::class_<JointVelocityMotion, JointVelocityWaypointMotion, std::shared_ptr<JointVelocityMotion>>(
      m, "JointVelocityMotion")
      .def(
          py::init<const Vector7d &, franka::Duration, RelativeDynamicsFactor>(),
          "target"_a,
          "duration"_a = franka::Duration(1000),
          "relative_dynamics_factor"_a = 1.0);

  py::class_<
      StopMotion<franka::JointVelocities>,
      Motion<franka::JointVelocities>,
      std::shared_ptr<StopMotion<franka::JointVelocities>>>(m, "JointVelocityStopMotion")
      .def(py::init<RelativeDynamicsFactor>(), "relative_dynamics_factor"_a = 1.0);
}
