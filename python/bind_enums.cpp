#include <pybind11/pybind11.h>

#include "docstrings.hpp"
#include "franky.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the '_a' literal
using namespace franky;

void bind_enums(py::module &m) {
  py::enum_<ReferenceType>(m, "ReferenceType", DOC(franky, ReferenceType))
      .value("Relative", ReferenceType::kRelative, DOC(franky, ReferenceType, kRelative))
      .value("Absolute", ReferenceType::kAbsolute, DOC(franky, ReferenceType, kAbsolute));

  py::enum_<franka::ControllerMode>(m, "ControllerMode", DOC(franka, ControllerMode))
      .value("JointImpedance", franka::ControllerMode::kJointImpedance)
      .value("CartesianImpedance", franka::ControllerMode::kCartesianImpedance);

  py::enum_<franka::RealtimeConfig>(m, "RealtimeConfig", DOC(franka, RealtimeConfig))
      .value("Enforce", franka::RealtimeConfig::kEnforce)
      .value("Ignore", franka::RealtimeConfig::kIgnore);

  py::enum_<ControlSignalType>(m, "ControlSignalType", DOC(franky, ControlSignalType))
      .value("Torques", ControlSignalType::Torques)
      .value("JointVelocities", ControlSignalType::JointVelocities)
      .value("JointPositions", ControlSignalType::JointPositions)
      .value("CartesianVelocities", ControlSignalType::CartesianVelocities)
      .value("CartesianPose", ControlSignalType::CartesianPose);

  py::enum_<franka::RobotMode>(m, "RobotMode", DOC(franka, RobotMode))
      .value("Other", franka::RobotMode::kOther)
      .value("Idle", franka::RobotMode::kIdle)
      .value("Move", franka::RobotMode::kMove)
      .value("Guiding", franka::RobotMode::kGuiding)
      .value("Reflex", franka::RobotMode::kReflex)
      .value("UserStopped", franka::RobotMode::kUserStopped)
      .value("AutomaticErrorRecovery", franka::RobotMode::kAutomaticErrorRecovery);

  py::enum_<FlipDirection>(m, "FlipDirection", DOC(franky, FlipDirection))
      .value("Negative", FlipDirection::kNegative, DOC(franky, FlipDirection, kNegative))
      .value("Neutral", FlipDirection::kNeutral, DOC(franky, FlipDirection, kNeutral))
      .value("Positive", FlipDirection::kPositive, DOC(franky, FlipDirection, kPositive));
}
