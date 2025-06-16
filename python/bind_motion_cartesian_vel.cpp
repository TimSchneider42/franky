#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "franky.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the '_a' literal
using namespace franky;

void bind_motion_cartesian_vel(py::module &m) {
  py::class_<VelocityWaypoint<RobotVelocity>>(m, "CartesianVelocityWaypoint")
      .def(
          py::init<>([](const RobotVelocity &target,
                        RelativeDynamicsFactor relative_dynamics_factor,
                        std::optional<franka::Duration>
                            minimum_time,
                        franka::Duration hold_target_duration,
                        std::optional<franka::Duration>
                            max_total_duration) {
            return VelocityWaypoint<RobotVelocity>{
                target, relative_dynamics_factor, minimum_time, hold_target_duration, max_total_duration};
          }),
          "target"_a,
          "relative_dynamics_factor"_a = 1.0,
          "minimum_time"_a = std::nullopt,
          "hold_target_duration"_a = franka::Duration(0),
          "max_total_duration"_a = std::nullopt)
      .def_readonly("target", &VelocityWaypoint<RobotVelocity>::target)
      .def_readonly("relative_dynamics_factor", &VelocityWaypoint<RobotVelocity>::relative_dynamics_factor)
      .def_readonly("minimum_time", &VelocityWaypoint<RobotVelocity>::minimum_time)
      .def_readonly("hold_target_duration", &VelocityWaypoint<RobotVelocity>::hold_target_duration)
      .def_readonly("max_total_duration", &VelocityWaypoint<RobotVelocity>::max_total_duration);

  py::class_<
      CartesianVelocityWaypointMotion,
      Motion<franka::CartesianVelocities>,
      std::shared_ptr<CartesianVelocityWaypointMotion>>(m, "CartesianVelocityWaypointMotion")
      .def(
          py::init<>([](const std::vector<VelocityWaypoint<RobotVelocity>> &waypoints,
                        const std::optional<Affine> &ee_frame = std::nullopt,
                        const RelativeDynamicsFactor &relative_dynamics_factor = 1.0) {
            return std::make_shared<CartesianVelocityWaypointMotion>(
                waypoints, relative_dynamics_factor, ee_frame.value_or(Affine::Identity()));
          }),
          "waypoints"_a,
          "ee_frame"_a = std::nullopt,
          "relative_dynamics_factor"_a = 1.0);

  py::class_<CartesianVelocityMotion, CartesianVelocityWaypointMotion, std::shared_ptr<CartesianVelocityMotion>>(
      m, "CartesianVelocityMotion")
      .def(
          py::init<>([](const RobotVelocity &target,
                        franka::Duration duration,
                        RelativeDynamicsFactor relative_dynamics_factor,
                        const std::optional<Affine> &ee_frame) {
            return std::make_shared<CartesianVelocityMotion>(
                target, duration, relative_dynamics_factor, ee_frame.value_or(Affine::Identity()));
          }),
          "target"_a,
          "duration"_a = franka::Duration(1000),
          "relative_dynamics_factor"_a = 1.0,
          "ee_frame"_a = std::nullopt);

  py::class_<
      StopMotion<franka::CartesianVelocities>,
      Motion<franka::CartesianVelocities>,
      std::shared_ptr<StopMotion<franka::CartesianVelocities>>>(m, "CartesianVelocityStopMotion")
      .def(py::init<RelativeDynamicsFactor>(), "relative_dynamics_factor"_a = 1.0);
}
