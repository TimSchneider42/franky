#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "franky.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the '_a' literal
using namespace franky;

void bind_motion_torque(py::module &m) {
  py::class_<ImpedanceMotion, Motion<franka::Torques>, std::shared_ptr<ImpedanceMotion>>(m, "ImpedanceMotion");

  py::class_<JointImpedanceMotion, Motion<franka::Torques>, std::shared_ptr<JointImpedanceMotion>>(
      m, "JointImpedanceMotion")
      .def(
          py::init<>([](const Vector7d &target,
                        std::optional<Vector7d> target_velocity,
                        std::optional<Vector7d> stiffness,
                        std::optional<Vector7d> damping,
                        std::optional<Vector7d> torque_feedforward,
                        std::optional<Vector7d> lower_joint_limits,
                        std::optional<Vector7d> upper_joint_limits,
                        bool compensate_coriolis,
                        double max_delta_tau,
                        double joint_limit_activation_distance,
                        double joint_limit_stiffness,
                        double joint_limit_damping,
                        double joint_limit_max_torque) {
            auto params = JointImpedanceMotion::Params{};
            if (stiffness.has_value()) params.stiffness = stiffness.value();
            if (damping.has_value()) params.damping = damping.value();
            if (torque_feedforward.has_value()) params.torque_feedforward = torque_feedforward.value();
            if (lower_joint_limits.has_value() && upper_joint_limits.has_value()) {
              params.joint_limit_repulsion_active = true;
              params.lower_joint_limits = lower_joint_limits.value();
              params.upper_joint_limits = upper_joint_limits.value();
            }
            params.compensate_coriolis = compensate_coriolis;
            params.max_delta_tau = max_delta_tau;
            params.joint_limit_activation_distance = joint_limit_activation_distance;
            params.joint_limit_stiffness = joint_limit_stiffness;
            params.joint_limit_damping = joint_limit_damping;
            params.joint_limit_max_torque = joint_limit_max_torque;

            if (target_velocity.has_value()) {
              return std::make_shared<JointImpedanceMotion>(target, target_velocity.value(), params);
            }
            return std::make_shared<JointImpedanceMotion>(target, params);
          }),
          "target"_a,
          "target_velocity"_a = std::nullopt,
          "stiffness"_a = std::nullopt,
          "damping"_a = std::nullopt,
          "torque_feedforward"_a = std::nullopt,
          "lower_joint_limits"_a = std::nullopt,
          "upper_joint_limits"_a = std::nullopt,
          "compensate_coriolis"_a = true,
          "max_delta_tau"_a = 1.0,
          "joint_limit_activation_distance"_a = 0.1,
          "joint_limit_stiffness"_a = 4.0,
          "joint_limit_damping"_a = 1.0,
          "joint_limit_max_torque"_a = 5.0)
      .def_property_readonly("target", &JointImpedanceMotion::target)
      .def_property_readonly("target_velocity", &JointImpedanceMotion::target_velocity);

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
