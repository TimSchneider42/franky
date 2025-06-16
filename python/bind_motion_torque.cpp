#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "franky.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the '_a' literal
using namespace franky;

void bind_motion_torque(py::module &m) {
  py::class_<ImpedanceMotion, Motion<franka::Torques>, std::shared_ptr<ImpedanceMotion>>(m, "ImpedanceMotion");

  py::class_<ExponentialImpedanceMotion, ImpedanceMotion, std::shared_ptr<ExponentialImpedanceMotion>>(
      m, "ExponentialImpedanceMotion")
      .def(
          py::init<>([](const Affine &target,
                        ReferenceType target_type,
                        double translational_stiffness,
                        double rotational_stiffness,
                        std::optional<std::array<std::optional<double>, 6>>
                            force_constraints,
                        double exponential_decay = 0.005) {
            Eigen::Vector<bool, 6> force_constraints_active = Eigen::Vector<bool, 6>::Zero();
            Eigen::Vector<double, 6> force_constraints_value;
            if (force_constraints.has_value()) {
              for (int i = 0; i < 6; i++) {
                force_constraints_value[i] = force_constraints.value()[i].value_or(NAN);
                force_constraints_active[i] = force_constraints.value()[i].has_value();
              }
            }
            return std::make_shared<ExponentialImpedanceMotion>(
                target,
                ExponentialImpedanceMotion::Params{
                    target_type,
                    translational_stiffness,
                    rotational_stiffness,
                    force_constraints_value,
                    force_constraints_active,
                    exponential_decay});
          }),
          "target"_a,
          py::arg_v("target_type", ReferenceType::kAbsolute, "_franky.ReferenceType.Absolute"),
          "translational_stiffness"_a = 2000,
          "rotational_stiffness"_a = 200,
          "force_constraints"_a = std::nullopt,
          "exponential_decay"_a = 0.005);

  py::class_<CartesianImpedanceMotion, ImpedanceMotion, std::shared_ptr<CartesianImpedanceMotion>>(
      m, "CartesianImpedanceMotion")
      .def(
          py::init<>([](const Affine &target,
                        franka::Duration duration,
                        ReferenceType target_type,
                        double translational_stiffness,
                        double rotational_stiffness,
                        std::optional<std::array<std::optional<double>, 6>>
                            force_constraints,
                        bool return_when_finished,
                        double finish_wait_factor) {
            Eigen::Vector<bool, 6> force_constraints_active = Eigen::Vector<bool, 6>::Zero();
            Eigen::Vector<double, 6> force_constraints_value;
            if (force_constraints.has_value()) {
              for (int i = 0; i < 6; i++) {
                force_constraints_value[i] = force_constraints.value()[i].value_or(NAN);
                force_constraints_active[i] = force_constraints.value()[i].has_value();
              }
            }
            return std::make_shared<CartesianImpedanceMotion>(
                target,
                duration,
                CartesianImpedanceMotion::Params{
                    target_type,
                    translational_stiffness,
                    rotational_stiffness,
                    force_constraints_value,
                    force_constraints_active,
                    return_when_finished,
                    finish_wait_factor});
          }),
          "target"_a,
          "duration"_a,
          py::arg_v("target_type", ReferenceType::kAbsolute, "_franky.ReferenceType.Absolute"),
          "translational_stiffness"_a = 2000,
          "rotational_stiffness"_a = 200,
          "force_constraints"_a = std::nullopt,
          "return_when_finished"_a = true,
          "finish_wait_factor"_a = 1.2);
}
