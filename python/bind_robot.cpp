#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "docstrings.hpp"
#include "franky.hpp"
#include "util.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the '_a' literal
using namespace franky;

namespace {

struct RobotDeleter {
  void operator()(Robot *robot) const {
    if (PyGILState_Check()) {
      py::gil_scoped_release release;
      delete robot;
    } else {
      delete robot;
    }
  }
};

using PythonRobotPtr = std::unique_ptr<Robot, RobotDeleter>;

}  // namespace

template <typename ControlSignalType>
void robotMove(
    Robot &robot, const std::shared_ptr<Motion<ControlSignalType>> &motion, bool async, bool limit_rate,
    double cutoff_frequency) {
  robot.move(motion, true, limit_rate, cutoff_frequency);
  if (!async) {
    auto future = std::async(std::launch::async, (bool (Robot::*)())&Robot::joinMotion, &robot);
    try {
      waitForFutureInterruptibly(future);
    } catch (const py::error_already_set &) {
      robot.stop();
      future.wait();
      throw;
    }
    future.get();
  }
}

constexpr const char *kMoveDoc =
    R"doc(Execute the given motion.

If a motion is already running, the new motion is queued and takes over seamlessly. The type of
control signal (e.g. joint positions, cartesian velocities, torques, ...) must not change while
the robot is in motion.

Args:
    motion: The motion to execute.
    asynchronous: Whether to return immediately instead of waiting for the motion to finish.
                  Asynchronous motions can be joined with join_motion.
    limit_rate: True if Franka's rate limiting should be activated. False by default.
                This could distort your motion!
    cutoff_frequency: Cutoff frequency for a first order low-pass filter applied on
                      the user commanded signal. Set to franka.kMaxCutoffFrequency to disable.)doc";

void bind_robot(py::module &m) {
  py::class_<DynamicsLimit<Vector7d>>(m, "VectorDynamicsLimit", DOC(franky, DynamicsLimit))
      .def(
          "set",
          &DynamicsLimit<Vector7d>::setFrom<Array<7>>,
          "value"_a,
          py::call_guard<py::gil_scoped_release>(),
          DOC(franky, DynamicsLimit, set))
      .def("get", &DynamicsLimit<Vector7d>::get, DOC(franky, DynamicsLimit, get))
      .def("__repr__", strFromStream<DynamicsLimit<Vector7d>>)
      .def_property_readonly(
          "max",
          [](const DynamicsLimit<Vector7d> &dynamics_limit) { return Vector7d(dynamics_limit.max); },
          DOC(franky, DynamicsLimit, max))
      .def_readonly("desc", &DynamicsLimit<Vector7d>::desc, DOC(franky, DynamicsLimit, desc));

  py::class_<DynamicsLimit<double>>(m, "DoubleDynamicsLimit", DOC(franky, DynamicsLimit))
      .def(
          "set",
          &DynamicsLimit<double>::set,
          "value"_a,
          py::call_guard<py::gil_scoped_release>(),
          DOC(franky, DynamicsLimit, set))
      .def("get", &DynamicsLimit<double>::get, DOC(franky, DynamicsLimit, get))
      .def("__repr__", strFromStream<DynamicsLimit<double>>)
      .def_readonly("max", &DynamicsLimit<double>::max, DOC(franky, DynamicsLimit, max))
      .def_readonly("desc", &DynamicsLimit<double>::desc, DOC(franky, DynamicsLimit, desc));

  py::class_<Gripper>(m, "Gripper", DOC(franky, Gripper))
      .def(
          py::init<const std::string &>(),
          "fci_hostname"_a,
          py::call_guard<py::gil_scoped_release>(),
          DOC(franka, Gripper, Gripper))
      .def(
          "grasp",
          &Gripper::grasp,
          "width"_a,
          "speed"_a,
          "force"_a,
          "epsilon_inner"_a = 0.005,
          "epsilon_outer"_a = 0.005,
          py::call_guard<py::gil_scoped_release>(),
          DOC(franka, Gripper, grasp))
      .def(
          "grasp_async",
          &Gripper::graspAsync,
          "width"_a,
          "speed"_a,
          "force"_a,
          "epsilon_inner"_a = 0.005,
          "epsilon_outer"_a = 0.005,
          py::call_guard<py::gil_scoped_release>(),
          DOC(franky, Gripper, graspAsync))
      .def(
          "move",
          &Gripper::move,
          "width"_a,
          "speed"_a,
          py::call_guard<py::gil_scoped_release>(),
          DOC(franka, Gripper, move))
      .def(
          "move_async",
          &Gripper::moveAsync,
          "width"_a,
          "speed"_a,
          py::call_guard<py::gil_scoped_release>(),
          DOC(franky, Gripper, moveAsync))
      .def("open", &Gripper::open, "speed"_a, py::call_guard<py::gil_scoped_release>(), DOC(franky, Gripper, open))
      .def(
          "open_async",
          &Gripper::openAsync,
          "speed"_a,
          py::call_guard<py::gil_scoped_release>(),
          DOC(franky, Gripper, openAsync))
      .def("homing", &Gripper::homing, py::call_guard<py::gil_scoped_release>(), DOC(franka, Gripper, homing))
      .def(
          "homing_async",
          &Gripper::homingAsync,
          py::call_guard<py::gil_scoped_release>(),
          DOC(franky, Gripper, homingAsync))
      .def("stop", &Gripper::stop, py::call_guard<py::gil_scoped_release>(), DOC(franka, Gripper, stop))
      .def("stop_async", &Gripper::stopAsync, py::call_guard<py::gil_scoped_release>(), DOC(franky, Gripper, stopAsync))
      .def_property_readonly(
          "state",
          py::cpp_function(&Gripper::state, py::call_guard<py::gil_scoped_release>()),
          DOC(franky, Gripper, state))
      .def_property_readonly(
          "server_version",
          [](const Gripper &gripper) { return static_cast<uint16_t>(gripper.serverVersion()); },
          DOC(franka, Gripper, serverVersion))
      .def_property_readonly(
          "width",
          py::cpp_function(&Gripper::width, py::call_guard<py::gil_scoped_release>()),
          DOC(franky, Gripper, width))
      .def_property_readonly(
          "is_grasped",
          py::cpp_function(&Gripper::is_grasped, py::call_guard<py::gil_scoped_release>()),
          DOC(franky, Gripper, is_grasped))
      .def_property_readonly(
          "max_width",
          py::cpp_function(&Gripper::max_width, py::call_guard<py::gil_scoped_release>()),
          DOC(franky, Gripper, max_width));

  py::class_<Robot, PythonRobotPtr>(m, "_RobotInternal", DOC(franky, Robot))
      .def(
          py::init<>([](const std::string &fci_hostname,
                        RelativeDynamicsFactor relative_dynamics_factor,
                        double default_torque_threshold,
                        double default_force_threshold,
                        franka::ControllerMode controller_mode,
                        franka::RealtimeConfig realtime_config,
                        double kalman_q_process_var,
                        double kalman_dq_process_var,
                        double kalman_ddq_process_var,
                        double kalman_control_process_var,
                        double kalman_q_obs_var,
                        double kalman_dq_obs_var,
                        double kalman_q_d_obs_var,
                        double kalman_dq_d_obs_var,
                        double kalman_ddq_d_obs_var,
                        double kalman_control_adaptation_rate) {
            return PythonRobotPtr(new Robot(
                fci_hostname,
                Robot::Params{
                    relative_dynamics_factor,
                    default_torque_threshold,
                    default_force_threshold,
                    controller_mode,
                    realtime_config,
                    kalman_q_process_var,
                    kalman_dq_process_var,
                    kalman_ddq_process_var,
                    kalman_control_process_var,
                    kalman_q_obs_var,
                    kalman_dq_obs_var,
                    kalman_q_d_obs_var,
                    kalman_dq_d_obs_var,
                    kalman_ddq_d_obs_var,
                    kalman_control_adaptation_rate}));
          }),
          R"doc(Connect to a Franka robot.

Args:
    fci_hostname: The hostname or IP address of the robot.
    relative_dynamics_factor: Global relative dynamics factor. The maximum velocity,
                              acceleration, and jerk of the robot are scaled by this factor.
    default_torque_threshold: Default torque threshold for the collision behavior [Nm].
    default_force_threshold: Default force threshold for the collision behavior [N].
    controller_mode: Default controller mode. See the libfranka documentation for details.
    realtime_config: Realtime configuration. If set to RealtimeConfig.Enforce, the connection
                     fails if no realtime kernel is available.
    kalman_q_process_var, ...: Noise variances of the Kalman filter franky uses to estimate the
                               robot state (the ``*_est`` fields of RobotState).
    kalman_control_adaptation_rate: Rate of adaptation of the estimated robot state to the
                                    desired robot state.)doc",
          "fci_hostname"_a,
          "relative_dynamics_factor"_a = 1.0,
          "default_torque_threshold"_a = 20.0,
          "default_force_threshold"_a = 30.0,
          py::arg_v(
              "controller_mode", franka::ControllerMode::kJointImpedance, "_franky.ControllerMode.JointImpedance"),
          py::arg_v("realtime_config", franka::RealtimeConfig::kEnforce, "_franky.RealtimeConfig.Enforce"),
          "kalman_q_process_var"_a = 0.0001,
          "kalman_dq_process_var"_a = 0.001,
          "kalman_ddq_process_var"_a = 0.1,
          "kalman_control_process_var"_a = 1.0,
          "kalman_q_obs_var"_a = 0.01,
          "kalman_dq_obs_var"_a = 0.1,
          "kalman_q_d_obs_var"_a = 0.0001,
          "kalman_dq_d_obs_var"_a = 0.0001,
          "kalman_ddq_d_obs_var"_a = 0.0001,
          "kalman_control_adaptation_rate"_a = 0.1,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "recover_from_errors",
          &Robot::recoverFromErrors,
          py::call_guard<py::gil_scoped_release>(),
          DOC(franky, Robot, recoverFromErrors))
      .def(
          "move",
          &robotMove<franka::CartesianPose>,
          "motion"_a,
          "asynchronous"_a = false,
          "limit_rate"_a = false,
          "cutoff_frequency"_a = franka::kDefaultCutoffFrequency,
          py::call_guard<py::gil_scoped_release>(),
          kMoveDoc)
      .def(
          "move",
          &robotMove<franka::CartesianVelocities>,
          "motion"_a,
          "asynchronous"_a = false,
          "limit_rate"_a = false,
          "cutoff_frequency"_a = franka::kDefaultCutoffFrequency,
          py::call_guard<py::gil_scoped_release>(),
          kMoveDoc)
      .def(
          "move",
          &robotMove<franka::JointPositions>,
          "motion"_a,
          "asynchronous"_a = false,
          "limit_rate"_a = false,
          "cutoff_frequency"_a = franka::kDefaultCutoffFrequency,
          py::call_guard<py::gil_scoped_release>(),
          kMoveDoc)
      .def(
          "move",
          &robotMove<franka::JointVelocities>,
          "motion"_a,
          "asynchronous"_a = false,
          "limit_rate"_a = false,
          "cutoff_frequency"_a = franka::kDefaultCutoffFrequency,
          py::call_guard<py::gil_scoped_release>(),
          kMoveDoc)
      .def(
          "move",
          &robotMove<franka::Torques>,
          "motion"_a,
          "asynchronous"_a = false,
          "limit_rate"_a = false,
          "cutoff_frequency"_a = franka::kDefaultCutoffFrequency,
          py::call_guard<py::gil_scoped_release>(),
          kMoveDoc)
      .def(
          "join_motion",
          [](Robot &robot, std::optional<double> timeout) {
            try {
              return interruptibleWait(
                  [&robot](const auto &wait_duration) { return robot.joinMotion(wait_duration); }, timeout);
            } catch (const py::error_already_set &) {
              robot.stop();
              robot.joinMotion();
              throw;
            }
          },
          "timeout"_a = std::nullopt,
          py::call_guard<py::gil_scoped_release>(),
          R"doc(Wait for the current motion to finish and throw any exception that occurred during the motion.

Args:
    timeout: Maximum time to wait [s]. If None, waits indefinitely.

Returns:
    True if the motion finished before the timeout expired, False otherwise.)doc")
      .def("poll_motion", &Robot::pollMotion, py::call_guard<py::gil_scoped_release>(), DOC(franky, Robot, pollMotion))
      .def(
          "set_collision_behavior",
          py::overload_cast<const ScalarOrArray<7> &, const ScalarOrArray<6> &>(&Robot::setCollisionBehavior),
          "torque_thresholds"_a,
          "force_thresholds"_a,
          py::call_guard<py::gil_scoped_release>(),
          DOC(franky, Robot, setCollisionBehavior))
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
          "upper_force_threshold"_a,
          py::call_guard<py::gil_scoped_release>(),
          DOC(franky, Robot, setCollisionBehavior_2))
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
          "upper_force_threshold_nominal"_a,
          py::call_guard<py::gil_scoped_release>(),
          DOC(franky, Robot, setCollisionBehavior_3))
      .def(
          "set_joint_impedance",
          &Robot::setJointImpedance,
          "K_theta"_a,
          py::call_guard<py::gil_scoped_release>(),
          DOC(franka, Robot, setJointImpedance))
      .def(
          "set_cartesian_impedance",
          &Robot::setCartesianImpedance,
          "K_x"_a,
          py::call_guard<py::gil_scoped_release>(),
          DOC(franka, Robot, setCartesianImpedance))
      .def(
          "set_guiding_mode",
          &Robot::setGuidingMode,
          "guiding_mode"_a,
          "elbow"_a,
          py::call_guard<py::gil_scoped_release>(),
          DOC(franka, Robot, setGuidingMode))
      .def("set_k", &Robot::setK, "EE_T_K"_a, py::call_guard<py::gil_scoped_release>(), DOC(franka, Robot, setK))
      .def("set_ee", &Robot::setEE, "NE_T_EE"_a, py::call_guard<py::gil_scoped_release>(), DOC(franka, Robot, setEE))
      .def(
          "set_load",
          &Robot::setLoad,
          "load_mass"_a,
          "F_x_Cload"_a,
          "load_inertia"_a,
          py::call_guard<py::gil_scoped_release>(),
          DOC(franka, Robot, setLoad))
      .def("stop", &Robot::stop, py::call_guard<py::gil_scoped_release>(), DOC(franka, Robot, stop))
      .def_property(
          "relative_dynamics_factor",
          py::cpp_function(&Robot::relative_dynamics_factor, py::call_guard<py::gil_scoped_release>()),
          py::cpp_function(&Robot::setRelativeDynamicsFactor, py::call_guard<py::gil_scoped_release>()),
          DOC(franky, Robot, relative_dynamics_factor))
      .def_property_readonly(
          "has_errors",
          py::cpp_function(&Robot::hasErrors, py::call_guard<py::gil_scoped_release>()),
          DOC(franky, Robot, hasErrors))
      .def_property_readonly(
          "current_pose",
          py::cpp_function(&Robot::currentPose, py::call_guard<py::gil_scoped_release>()),
          DOC(franky, Robot, currentPose))
      .def_property_readonly(
          "current_cartesian_velocity",
          py::cpp_function(&Robot::currentCartesianVelocity, py::call_guard<py::gil_scoped_release>()),
          DOC(franky, Robot, currentCartesianVelocity))
      .def_property_readonly(
          "current_cartesian_state",
          py::cpp_function(&Robot::currentCartesianState, py::call_guard<py::gil_scoped_release>()),
          DOC(franky, Robot, currentCartesianState))
      .def_property_readonly(
          "current_joint_state",
          py::cpp_function(&Robot::currentJointState, py::call_guard<py::gil_scoped_release>()),
          DOC(franky, Robot, currentJointState))
      .def_property_readonly(
          "current_joint_velocities",
          py::cpp_function(&Robot::currentJointVelocities, py::call_guard<py::gil_scoped_release>()),
          DOC(franky, Robot, currentJointVelocities))
      .def_property_readonly(
          "current_joint_positions",
          py::cpp_function(&Robot::currentJointPositions, py::call_guard<py::gil_scoped_release>()),
          DOC(franky, Robot, currentJointPositions))
      .def_property_readonly(
          "state", py::cpp_function(&Robot::state, py::call_guard<py::gil_scoped_release>()), DOC(franky, Robot, state))
      .def_property_readonly(
          "is_in_control",
          py::cpp_function(&Robot::is_in_control, py::call_guard<py::gil_scoped_release>()),
          DOC(franky, Robot, is_in_control))
      .def_property_readonly("fci_hostname", &Robot::fci_hostname, DOC(franky, Robot, fci_hostname))
      .def_property_readonly(
          "current_control_signal_type",
          py::cpp_function(&Robot::current_control_signal_type, py::call_guard<py::gil_scoped_release>()),
          DOC(franky, Robot, current_control_signal_type))
      .def_property_readonly("model", &Robot::model, DOC(franky, Robot, model))
#ifdef FRANKA_0_15
      .def_property_readonly("model_urdf", &Robot::model_urdf, DOC(franky, Robot, model_urdf))
#endif
      .def_readonly(
          "translation_velocity_limit",
          &Robot::translation_velocity_limit,
          DOC(franky, Robot, translation_velocity_limit))
      .def_readonly(
          "rotation_velocity_limit", &Robot::rotation_velocity_limit, DOC(franky, Robot, rotation_velocity_limit))
      .def_readonly("elbow_velocity_limit", &Robot::elbow_velocity_limit, DOC(franky, Robot, elbow_velocity_limit))
      .def_readonly(
          "translation_acceleration_limit",
          &Robot::translation_acceleration_limit,
          DOC(franky, Robot, translation_acceleration_limit))
      .def_readonly(
          "rotation_acceleration_limit",
          &Robot::rotation_acceleration_limit,
          DOC(franky, Robot, rotation_acceleration_limit))
      .def_readonly(
          "elbow_acceleration_limit", &Robot::elbow_acceleration_limit, DOC(franky, Robot, elbow_acceleration_limit))
      .def_readonly(
          "translation_jerk_limit", &Robot::translation_jerk_limit, DOC(franky, Robot, translation_jerk_limit))
      .def_readonly("rotation_jerk_limit", &Robot::rotation_jerk_limit, DOC(franky, Robot, rotation_jerk_limit))
      .def_readonly("elbow_jerk_limit", &Robot::elbow_jerk_limit, DOC(franky, Robot, elbow_jerk_limit))
      .def_readonly("joint_velocity_limit", &Robot::joint_velocity_limit, DOC(franky, Robot, joint_velocity_limit))
      .def_readonly(
          "joint_acceleration_limit", &Robot::joint_acceleration_limit, DOC(franky, Robot, joint_acceleration_limit))
      .def_readonly("joint_jerk_limit", &Robot::joint_jerk_limit, DOC(franky, Robot, joint_jerk_limit))
      .def_readonly_static(
          "degrees_of_freedom", &Robot::degrees_of_freedoms, "Number of degrees of freedom of the robot.")
      .def_readonly_static("control_rate", &Robot::control_rate, "Control rate of the robot [s].");
}
