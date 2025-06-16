#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "franky.hpp"
#include "sequential_executor.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the '_a' literal
using namespace franky;

SequentialExecutor callback_executor;

template <typename ControlSignalType>
void mkMotionAndReactionClasses(py::module_ &m, const std::string &control_signal_name) {
  py::class_<Motion<ControlSignalType>, std::shared_ptr<Motion<ControlSignalType>>> motion_class(
      m, ("Base" + control_signal_name + "Motion").c_str());
  py::class_<Reaction<ControlSignalType>, std::shared_ptr<Reaction<ControlSignalType>>> reaction_class(
      m, (control_signal_name + "Reaction").c_str());

  motion_class.def_property_readonly("reactions", &Motion<ControlSignalType>::reactions)
      .def("add_reaction", &Motion<ControlSignalType>::addReaction)
      .def(
          "register_callback",
          [](Motion<ControlSignalType> &motion, const typename Motion<ControlSignalType>::CallbackType &callback) {
            motion.registerCallback([callback](
                                        const RobotState &robot_state,
                                        franka::Duration time_step,
                                        franka::Duration rel_time,
                                        franka::Duration abs_time,
                                        const ControlSignalType &control_signal) {
              callback_executor.add([callback, robot_state, time_step, rel_time, abs_time, control_signal]() {
                try {
                  callback(robot_state, time_step, rel_time, abs_time, control_signal);
                } catch (const py::error_already_set &e) {
                  std::cerr << "Error in callback: " << std::endl;
                  py::gil_scoped_acquire gil_acquire;
                  PyErr_Print();
                }
              });
            });
          },
          "callback"_a);

  reaction_class
      .def(
          py::init<const Condition &, std::shared_ptr<Motion<ControlSignalType>>>(),
          "condition"_a,
          "motion"_a = nullptr)
      .def(
          "register_callback",
          [](Reaction<ControlSignalType> &reaction,
             const std::function<void(const RobotState &, franka::Duration, franka::Duration)> &callback) {
            reaction.registerCallback(
                [callback](const RobotState &robot_state, franka::Duration rel_time, franka::Duration abs_time) {
                  callback_executor.add([callback, robot_state, rel_time, abs_time]() {
                    try {
                      callback(robot_state, rel_time, abs_time);
                    } catch (const py::error_already_set &e) {
                      py::gil_scoped_acquire gil_acquire;
                      std::cerr << "Error in callback: ";
                      PyErr_Print();
                    }
                  });
                });
          },
          "callback"_a);
}

void bind_reactions(py::module &m) {
  py::class_<Condition>(m, "Condition")
      .def(py::init<bool>(), "constant_value"_a)
      .def("__invert__", py::overload_cast<const Condition &>(&operator!), py::is_operator())
      .def(py::self == py::self)
      .def(py::self == bool())
      .def(py::self != py::self)
      .def(py::self != bool())
      .def("__and__", py::overload_cast<const Condition &, const Condition &>(&operator&&), py::is_operator())
      .def(
          "__and__", [](const Condition &condition, bool constant) { return condition && constant; }, py::is_operator())
      .def(
          "__rand__",
          [](bool constant, const Condition &condition) { return constant && condition; },
          py::is_operator())
      .def("__or__", py::overload_cast<const Condition &, const Condition &>(&operator||), py::is_operator())
      .def(
          "__or__", [](const Condition &condition, bool constant) { return condition || constant; }, py::is_operator())
      .def(
          "__ror__", [](bool constant, const Condition &condition) { return constant || condition; }, py::is_operator())
      .def("__repr__", &Condition::repr);
  py::implicitly_convertible<bool, Condition>();

  py::class_<Measure>(m, "Measure")
      .def_property_readonly_static("FORCE_X", [](py::object) { return Measure::ForceX(); })
      .def_property_readonly_static("FORCE_Y", [](py::object) { return Measure::ForceY(); })
      .def_property_readonly_static("FORCE_Z", [](py::object) { return Measure::ForceZ(); })
      .def_property_readonly_static("REL_TIME", [](py::object) { return Measure::RelTime(); })
      .def_property_readonly_static("ABS_TIME", [](py::object) { return Measure::AbsTime(); })
      .def(-py::self)
      .def(py::self == py::self)
      .def(py::self == double_t())
      .def(py::self != py::self)
      .def(py::self != double_t())
      .def(py::self > py::self)
      .def(py::self > double_t())
      .def(py::self >= py::self)
      .def(py::self >= double_t())
      .def(py::self < py::self)
      .def(py::self < double_t())
      .def(py::self <= py::self)
      .def(py::self <= double_t())
      .def(py::self + py::self)
      .def(py::self + double_t())
      .def(double_t() + py::self)
      .def(py::self - py::self)
      .def(py::self - double_t())
      .def(double_t() - py::self)
      .def(py::self * py::self)
      .def(py::self * double_t())
      .def(double_t() * py::self)
      .def(py::self / py::self)
      .def(py::self / double_t())
      .def(double_t() / py::self)
      .def("__pow__", py::overload_cast<const Measure &, const Measure &>(&measure_pow), py::is_operator())
      .def(
          "__pow__",
          [](const Measure &measure, double constant) { return measure_pow(measure, constant); },
          py::is_operator())
      .def(
          "__rpow__",
          [](const Measure &measure, double constant) { return measure_pow(constant, measure); },
          py::is_operator())
      .def("__repr__", &Measure::repr);

  mkMotionAndReactionClasses<franka::Torques>(m, "Torque");
  mkMotionAndReactionClasses<franka::JointVelocities>(m, "JointVelocity");
  mkMotionAndReactionClasses<franka::JointPositions>(m, "JointPosition");
  mkMotionAndReactionClasses<franka::CartesianVelocities>(m, "CartesianVelocity");
  mkMotionAndReactionClasses<franka::CartesianPose>(m, "CartesianPose");
}
