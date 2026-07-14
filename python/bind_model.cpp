#include <franka/model.h>
#include <franka/robot_state.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "docstrings.hpp"
#include "franky.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the '_a' literal
using namespace franky;

void bind_model(py::module &m) {
  py::enum_<franka::Frame>(m, "Frame", DOC(franka, Frame))
      .value("Joint1", franka::Frame::kJoint1)
      .value("Joint2", franka::Frame::kJoint2)
      .value("Joint3", franka::Frame::kJoint3)
      .value("Joint4", franka::Frame::kJoint4)
      .value("Joint5", franka::Frame::kJoint5)
      .value("Joint6", franka::Frame::kJoint6)
      .value("Joint7", franka::Frame::kJoint7)
      .value("Flange", franka::Frame::kFlange)
      .value("EndEffector", franka::Frame::kEndEffector)
      .value("Stiffness", franka::Frame::kStiffness);

  py::class_<Model, std::shared_ptr<Model>>(m, "Model", DOC(franky, Model))
      .def(
          "pose",
          py::overload_cast<franka::Frame, const RobotState &>(&Model::pose, py::const_),
          "frame"_a,
          "state"_a,
          DOC(franky, Model, pose))
      .def(
          "pose",
          py::overload_cast<franka::Frame, const Vector7d &, const Affine &, const Affine &>(&Model::pose, py::const_),
          "frame"_a,
          "q"_a,
          "F_T_EE"_a,
          "EE_T_K"_a,
          DOC(franky, Model, pose_2))

      .def(
          "body_jacobian",
          py::overload_cast<franka::Frame, const RobotState &>(&Model::bodyJacobian, py::const_),
          "frame"_a,
          "state"_a,
          DOC(franky, Model, bodyJacobian))
      .def(
          "body_jacobian",
          py::overload_cast<franka::Frame, const Vector7d &, const Affine &, const Affine &>(
              &Model::bodyJacobian, py::const_),
          "frame"_a,
          "q"_a,
          "F_T_EE"_a,
          "EE_T_K"_a,
          DOC(franky, Model, bodyJacobian_2))

      .def(
          "zero_jacobian",
          py::overload_cast<franka::Frame, const RobotState &>(&Model::zeroJacobian, py::const_),
          "frame"_a,
          "state"_a,
          DOC(franky, Model, zeroJacobian))
      .def(
          "zero_jacobian",
          py::overload_cast<franka::Frame, const Vector7d &, const Affine &, const Affine &>(
              &Model::zeroJacobian, py::const_),
          "frame"_a,
          "q"_a,
          "F_T_EE"_a,
          "EE_T_K"_a,
          DOC(franky, Model, zeroJacobian_2))

      .def("mass", py::overload_cast<const RobotState &>(&Model::mass, py::const_), "state"_a, DOC(franky, Model, mass))
      .def(
          "mass",
          py::overload_cast<const Vector7d &, const Eigen::Matrix3d &, double, const Eigen::Vector3d &>(
              &Model::mass, py::const_),
          "q"_a,
          "I_total"_a,
          "m_total"_a,
          "F_x_Ctotal"_a,
          DOC(franky, Model, mass_2))

      .def(
          "coriolis",
          py::overload_cast<const RobotState &>(&Model::coriolis, py::const_),
          "state"_a,
          DOC(franky, Model, coriolis))
      .def(
          "coriolis",
          py::overload_cast<
              const Vector7d &,
              const Vector7d &,
              const Eigen::Matrix3d &,
              double,
              const Eigen::Vector3d &>(&Model::coriolis, py::const_),
          "q"_a,
          "dq"_a,
          "I_total"_a,
          "m_total"_a,
          "F_x_Ctotal"_a,
          DOC(franky, Model, coriolis_2))

      .def(
          "gravity",
          py::overload_cast<const RobotState &, const Eigen::Vector3d &>(&Model::gravity, py::const_),
          "state"_a,
          "gravity_earth"_a,
          DOC(franky, Model, gravity))
      .def(
          "gravity",
          py::overload_cast<const RobotState &>(&Model::gravity, py::const_),
          "state"_a,
          DOC(franky, Model, gravity_2))
      .def(
          "gravity",
          py::overload_cast<const Vector7d &, double, const Eigen::Vector3d &, const Eigen::Vector3d &>(
              &Model::gravity, py::const_),
          "q"_a,
          "m_total"_a,
          "F_x_Ctotal"_a,
          "gravity_earth"_a = Eigen::Vector3d{0., 0., -9.81},
          DOC(franky, Model, gravity_3));
}
