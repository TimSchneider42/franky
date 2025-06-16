#include <Python.h>
#include <franka/robot_state.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the '_a' literal

void bind_enums(py::module &);
void bind_errors(py::module &);
void bind_misc(py::module &);
void bind_model(py::module &);
void bind_motion_cartesian_pos(py::module &);
void bind_motion_cartesian_vel(py::module &);
void bind_motion_joint_pos(py::module &);
void bind_motion_joint_vel(py::module &);
void bind_motion_torque(py::module &);
void bind_reactions(py::module &);
void bind_robot(py::module &);
void bind_robot_state(py::module &);
void bind_state_repr(py::module &);

PYBIND11_MODULE(_franky, m) {
  m.doc() = "High-Level Control Library for Franka Robots";

  bind_misc(m);
  bind_enums(m);
  bind_errors(m);
  bind_state_repr(m);
  bind_robot_state(m);
  bind_reactions(m);
  bind_model(m);
  bind_motion_cartesian_pos(m);
  bind_motion_cartesian_vel(m);
  bind_motion_joint_pos(m);
  bind_motion_joint_vel(m);
  bind_motion_torque(m);
  bind_robot(m);
}
