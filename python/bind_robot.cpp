#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "franky.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the '_a' literal
using namespace franky;

template <typename ControlSignalType>
void robotMove(Robot &robot, const std::shared_ptr<Motion<ControlSignalType>> &motion, bool async) {
  robot.move(motion, true);
  if (!async) {
    auto future = std::async(std::launch::async, (bool(Robot::*)()) & Robot::joinMotion, &robot);
    // Check if python wants to terminate every 100 ms
    bool python_terminating = false;
    while (future.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout) {
      {
        py::gil_scoped_acquire gil_acquire;
        python_terminating = Py_IsInitialized() && PyErr_CheckSignals() == -1;
      }
      if (python_terminating) {
        robot.stop();
        future.wait();
        py::gil_scoped_acquire gil_acquire;
        throw py::error_already_set();
      }
    }
    future.get();
  }
}

void bind_robot(py::module &m) {
  py::class_<DynamicsLimit<std::array<double, 7>>>(m, "VectorDynamicsLimit")
      .def("set", &DynamicsLimit<std::array<double, 7>>::setFrom<Array<7>>, "value"_a)
      .def("get", &DynamicsLimit<std::array<double, 7>>::getAs<Vector7d>)
      .def_property_readonly(
          "max",
          [](const DynamicsLimit<std::array<double, 7>> &dynamics_limit) {
            return Vector7d::Map(dynamics_limit.max.data());
          })
      .def_readonly("desc", &DynamicsLimit<std::array<double, 7>>::desc);

  py::class_<DynamicsLimit<double>>(m, "DoubleDynamicsLimit")
      .def("set", &DynamicsLimit<double>::set, "value"_a)
      .def("get", &DynamicsLimit<double>::get)
      .def_readonly("max", &DynamicsLimit<double>::max)
      .def_readonly("desc", &DynamicsLimit<double>::desc);

  py::class_<Gripper>(m, "Gripper")
      .def(py::init<const std::string &>(), "fci_hostname"_a)
      .def(
          "grasp",
          &Gripper::grasp,
          "width"_a,
          "speed"_a,
          "force"_a,
          "epsilon_inner"_a = 0.005,
          "epsilon_outer"_a = 0.005,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "grasp_async",
          &Gripper::graspAsync,
          "width"_a,
          "speed"_a,
          "force"_a,
          "epsilon_inner"_a = 0.005,
          "epsilon_outer"_a = 0.005,
          py::call_guard<py::gil_scoped_release>())
      .def("move", &Gripper::move, "width"_a, "speed"_a, py::call_guard<py::gil_scoped_release>())
      .def("move_async", &Gripper::moveAsync, "width"_a, "speed"_a, py::call_guard<py::gil_scoped_release>())
      .def("open", &Gripper::open, "speed"_a, py::call_guard<py::gil_scoped_release>())
      .def("open_async", &Gripper::openAsync, "speed"_a, py::call_guard<py::gil_scoped_release>())
      .def("homing", &Gripper::homing, py::call_guard<py::gil_scoped_release>())
      .def("homing_async", &Gripper::homingAsync, py::call_guard<py::gil_scoped_release>())
      .def("stop", &Gripper::stop, py::call_guard<py::gil_scoped_release>())
      .def("stop_async", &Gripper::stopAsync, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("state", &Gripper::state)
      .def_property_readonly("server_version", (uint16_t(Gripper::*)()) & Gripper::serverVersion)
      .def_property_readonly("width", &Gripper::width)
      .def_property_readonly("is_grasped", &Gripper::is_grasped)
      .def_property_readonly("max_width", &Gripper::max_width);

  py::class_<Robot>(m, "RobotInternal")
      .def(
          py::init<>([](const std::string &fci_hostname,
                        RelativeDynamicsFactor relative_dynamics_factor,
                        double default_torque_threshold,
                        double default_force_threshold,
                        franka::ControllerMode controller_mode,
                        franka::RealtimeConfig realtime_config) {
            return std::make_unique<Robot>(
                fci_hostname,
                Robot::Params{
                    relative_dynamics_factor,
                    default_torque_threshold,
                    default_force_threshold,
                    controller_mode,
                    realtime_config});
          }),
          "fci_hostname"_a,
          "relative_dynamics_factor"_a = 1.0,
          "default_torque_threshold"_a = 20.0,
          "default_force_threshold"_a = 30.0,
          py::arg_v(
              "controller_mode", franka::ControllerMode::kJointImpedance, "_franky.ControllerMode.JointImpedance"),
          py::arg_v("realtime_config", franka::RealtimeConfig::kEnforce, "_franky.RealtimeConfig.Enforce"))
      .def("recover_from_errors", &Robot::recoverFromErrors)
      .def(
          "move",
          &robotMove<franka::CartesianPose>,
          "motion"_a,
          "asynchronous"_a = false,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "move",
          &robotMove<franka::CartesianVelocities>,
          "motion"_a,
          "asynchronous"_a = false,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "move",
          &robotMove<franka::JointPositions>,
          "motion"_a,
          "asynchronous"_a = false,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "move",
          &robotMove<franka::JointVelocities>,
          "motion"_a,
          "asynchronous"_a = false,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "move",
          &robotMove<franka::Torques>,
          "motion"_a,
          "asynchronous"_a = false,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "join_motion",
          [](Robot &robot, std::optional<double> timeout) {
            if (timeout.has_value())
              return robot.joinMotion<double, std::ratio<1>>(
                  std::chrono::duration<double, std::ratio<1>>(timeout.value()));
            return robot.joinMotion();
          },
          "timeout"_a = std::nullopt,
          py::call_guard<py::gil_scoped_release>())
      .def("poll_motion", &Robot::pollMotion)
      .def(
          "set_collision_behavior",
          py::overload_cast<const ScalarOrArray<7> &, const ScalarOrArray<6> &>(&Robot::setCollisionBehavior),
          "torque_thresholds"_a,
          "force_thresholds"_a)
      .def(
          "set_collision_behavior",
          py::overload_cast<
              const ScalarOrArray<7> &,
              const ScalarOrArray<7> &,
              const ScalarOrArray<6> &,
              const ScalarOrArray<6> &>(&Robot::setCollisionBehavior),
          "lower_torque_threshold"_a,
          "upper_torque_threshold"_a,
          "lower_force_threshold"_a,
          "upper_force_threshold"_a)
      .def(
          "set_collision_behavior",
          py::overload_cast<
              const ScalarOrArray<7> &,
              const ScalarOrArray<7> &,
              const ScalarOrArray<7> &,
              const ScalarOrArray<7> &,
              const ScalarOrArray<6> &,
              const ScalarOrArray<6> &,
              const ScalarOrArray<6> &,
              const ScalarOrArray<6> &>(&Robot::setCollisionBehavior),
          "lower_torque_threshold_acceleration"_a,
          "upper_torque_threshold_acceleration"_a,
          "lower_torque_threshold_nominal"_a,
          "upper_torque_threshold_nominal"_a,
          "lower_force_threshold_acceleration"_a,
          "upper_force_threshold_acceleration"_a,
          "lower_force_threshold_nominal"_a,
          "upper_force_threshold_nominal"_a)
      .def("set_joint_impedance", &Robot::setJointImpedance, "K_theta"_a)
      .def("set_cartesian_impedance", &Robot::setCartesianImpedance, "K_x"_a)
      .def("set_guiding_mode", &Robot::setGuidingMode, "guiding_mode"_a, "elbow"_a)
      .def("set_k", &Robot::setK, "EE_T_K"_a)
      .def("set_ee", &Robot::setEE, "NE_T_EE"_a)
      .def("set_load", &Robot::setLoad, "load_mass"_a, "F_x_Cload"_a, "load_inertia"_a)
      .def("stop", &Robot::stop)
      .def_property("relative_dynamics_factor", &Robot::relative_dynamics_factor, &Robot::setRelativeDynamicsFactor)
      .def_property_readonly("has_errors", &Robot::hasErrors)
      .def_property_readonly("current_pose", &Robot::currentPose)
      .def_property_readonly("current_cartesian_velocity", &Robot::currentCartesianVelocity)
      .def_property_readonly("current_cartesian_state", &Robot::currentCartesianState)
      .def_property_readonly("current_joint_state", &Robot::currentJointState)
      .def_property_readonly("current_joint_velocities", &Robot::currentJointVelocities)
      .def_property_readonly("current_joint_positions", &Robot::currentJointPositions)
      .def_property_readonly("state", &Robot::state)
      .def_property_readonly("is_in_control", &Robot::is_in_control)
      .def_property_readonly("fci_hostname", &Robot::fci_hostname)
      .def_property_readonly("current_control_signal_type", &Robot::current_control_signal_type)
      .def_readonly("translation_velocity_limit", &Robot::translation_velocity_limit, "[m/s]")
      .def_readonly("rotation_velocity_limit", &Robot::rotation_velocity_limit, "[rad/s]")
      .def_readonly("elbow_velocity_limit", &Robot::elbow_velocity_limit, "[rad/s]")
      .def_readonly("translation_acceleration_limit", &Robot::translation_acceleration_limit, "[m/s²]")
      .def_readonly("rotation_acceleration_limit", &Robot::rotation_acceleration_limit, "[rad/s²]")
      .def_readonly("elbow_acceleration_limit", &Robot::elbow_acceleration_limit, "[rad/s²]")
      .def_readonly("translation_jerk_limit", &Robot::translation_jerk_limit, "[m/s³]")
      .def_readonly("rotation_jerk_limit", &Robot::rotation_jerk_limit, "[rad/s³]")
      .def_readonly("elbow_jerk_limit", &Robot::elbow_jerk_limit, "[rad/s³]")
      .def_readonly("joint_velocity_limit", &Robot::joint_velocity_limit, "[rad/s]")
      .def_readonly("joint_acceleration_limit", &Robot::joint_acceleration_limit, "[rad/s²]")
      .def_readonly("joint_jerk_limit", &Robot::joint_jerk_limit, "[rad/s^3]")
      .def_readonly_static("degrees_of_freedom", &Robot::degrees_of_freedoms)
      .def_readonly_static("control_rate", &Robot::control_rate, "[s]")
      .def_static("forward_kinematics", &Robot::forwardKinematics, "q"_a);
  //  .def_static("inverse_kinematics", &Robot::inverseKinematics, "target"_a, "q0"_a);
}