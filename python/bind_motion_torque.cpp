#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "franky.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the '_a' literal
using namespace franky;

namespace {
CartesianReference toCartesianReference(const Affine &target, const std::optional<Twist> &target_twist) {
  CartesianReference reference;
  reference.target = target;
  reference.target_twist = target_twist;
  return reference;
}

JointReference toJointReference(
    const Vector7d &position,
    const std::optional<Vector7d> &velocity,
    const std::optional<Vector7d> &torque_feedforward) {
  JointReference reference;
  reference.q = position;
  if (velocity.has_value()) reference.dq = velocity.value();
  if (torque_feedforward.has_value()) reference.tau_ff = torque_feedforward.value();
  return reference;
}

JointImpedanceParams makeJointImpedanceParams(
    const std::optional<Vector7d> &stiffness,
    const std::optional<Vector7d> &damping,
    const std::optional<Vector7d> &constant_torque_offset,
    const std::optional<Vector7d> &lower_joint_limits,
    const std::optional<Vector7d> &upper_joint_limits,
    bool compensate_coriolis,
    double max_delta_tau,
    double joint_limit_activation_distance,
    double joint_limit_stiffness,
    double joint_limit_damping,
    double joint_limit_max_torque) {
  auto params = JointImpedanceParams{};
  if (stiffness.has_value()) params.stiffness = stiffness.value();
  if (damping.has_value()) params.damping = damping.value();
  if (constant_torque_offset.has_value()) params.constant_torque_offset = constant_torque_offset.value();
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
  return params;
}

CartesianImpedanceBase::Params makeCartesianImpedanceParams(
    double translational_stiffness,
    double rotational_stiffness,
    const std::optional<std::array<std::optional<double>, 6>> &force_constraints,
    const std::optional<Vector7d> &nullspace_target,
    double nullspace_stiffness,
    double max_delta_tau,
    const std::optional<Vector7d> &lower_joint_limits,
    const std::optional<Vector7d> &upper_joint_limits,
    double joint_limit_activation_distance,
    double joint_limit_stiffness,
    double joint_limit_damping,
    double joint_limit_max_torque) {
  auto params = CartesianImpedanceBase::Params{};
  params.translational_stiffness = translational_stiffness;
  params.rotational_stiffness = rotational_stiffness;
  params.nullspace_target = nullspace_target;
  params.nullspace_stiffness = nullspace_stiffness;
  params.max_delta_tau = max_delta_tau;
  params.joint_limit_activation_distance = joint_limit_activation_distance;
  params.joint_limit_stiffness = joint_limit_stiffness;
  params.joint_limit_damping = joint_limit_damping;
  params.joint_limit_max_torque = joint_limit_max_torque;

  Eigen::Vector<bool, 6> force_constraints_active = Eigen::Vector<bool, 6>::Zero();
  Eigen::Vector<double, 6> force_constraints_value;
  if (force_constraints.has_value()) {
    for (int i = 0; i < 6; i++) {
      force_constraints_value[i] = force_constraints.value()[i].value_or(NAN);
      force_constraints_active[i] = force_constraints.value()[i].has_value();
    }
  }
  params.force_constraints = force_constraints_value;
  params.force_constraints_active = force_constraints_active;
  if (lower_joint_limits.has_value() && upper_joint_limits.has_value()) {
    params.joint_limit_repulsion_active = true;
    params.lower_joint_limits = lower_joint_limits.value();
    params.upper_joint_limits = upper_joint_limits.value();
  }
  return params;
}
}  // namespace

void bind_motion_torque(py::module &m) {
  auto cartesian_impedance_base = py::class_<CartesianImpedanceBase, Motion<franka::Torques>, std::shared_ptr<CartesianImpedanceBase>>(
      m, "CartesianImpedanceBase");
  m.attr("ImpedanceMotion") = cartesian_impedance_base;

  py::class_<JointReferenceHandle, std::shared_ptr<JointReferenceHandle>>(m, "JointReferenceHandle")
      .def(py::init<>())
      .def(
          "set",
          [](JointReferenceHandle &handle,
             const Vector7d &position,
             std::optional<Vector7d> velocity,
             std::optional<Vector7d> torque_feedforward) {
            handle.set(toJointReference(position, velocity, torque_feedforward));
          },
          R"doc(Update the current joint reference for a running JointImpedanceTrackingMotion.

The provided torque_feedforward is added on top of any constant_torque_offset configured on the motion.)doc",
          "position"_a,
          "velocity"_a = std::nullopt,
          "torque_feedforward"_a = std::nullopt)
      .def("clear", &JointReferenceHandle::clear)
      .def_property_readonly("has_reference", &JointReferenceHandle::hasReference);

  py::class_<CartesianReferenceHandle, std::shared_ptr<CartesianReferenceHandle>>(m, "CartesianReferenceHandle")
      .def(py::init<>())
      .def(
          "set",
          [](CartesianReferenceHandle &handle, const Affine &target, std::optional<Twist> target_twist) {
            handle.set(toCartesianReference(target, target_twist));
          },
          R"doc(Update the current Cartesian reference for a running CartesianImpedanceTrackingMotion.

If target_twist is provided, it is interpreted as the desired end-effector twist in the base frame and the damping term acts on twist error rather than motion relative to zero.)doc",
          "target"_a,
          "target_twist"_a = std::nullopt)
      .def("clear", &CartesianReferenceHandle::clear)
      .def_property_readonly("has_reference", &CartesianReferenceHandle::hasReference);

  py::class_<JointImpedanceMotion, Motion<franka::Torques>, std::shared_ptr<JointImpedanceMotion>>(
      m, "JointImpedanceMotion")
      .def(
          py::init<>([](const Vector7d &target,
                        std::optional<Vector7d> target_velocity,
                        std::optional<Vector7d> stiffness,
                        std::optional<Vector7d> damping,
                        std::optional<Vector7d> constant_torque_offset,
                        std::optional<Vector7d> lower_joint_limits,
                        std::optional<Vector7d> upper_joint_limits,
                        bool compensate_coriolis,
                        double max_delta_tau,
                        double joint_limit_activation_distance,
                        double joint_limit_stiffness,
                        double joint_limit_damping,
                        double joint_limit_max_torque) {
            auto params = makeJointImpedanceParams(
                stiffness,
                damping,
                constant_torque_offset,
                lower_joint_limits,
                upper_joint_limits,
                compensate_coriolis,
                max_delta_tau,
                joint_limit_activation_distance,
                joint_limit_stiffness,
                joint_limit_damping,
                joint_limit_max_torque);

            const Vector7d target_vector = target;
            if (target_velocity.has_value()) {
              return std::make_shared<JointImpedanceMotion>(
                  target_vector, target_velocity.value(), params);
            }
            return std::make_shared<JointImpedanceMotion>(target_vector, params);
          }),
          "target"_a,
          "target_velocity"_a = std::nullopt,
          "stiffness"_a = std::nullopt,
          "damping"_a = std::nullopt,
          "constant_torque_offset"_a = std::nullopt,
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

  py::class_<JointImpedanceTrackingMotion, Motion<franka::Torques>, std::shared_ptr<JointImpedanceTrackingMotion>>(
      m, "JointImpedanceTrackingMotion")
      .def(
          py::init<>([](const std::shared_ptr<JointReferenceHandle> &reference_handle,
                        std::optional<Vector7d> stiffness,
                        std::optional<Vector7d> damping,
                        std::optional<Vector7d> constant_torque_offset,
                        std::optional<Vector7d> lower_joint_limits,
                        std::optional<Vector7d> upper_joint_limits,
                        bool compensate_coriolis,
                        double max_delta_tau,
                        double joint_limit_activation_distance,
                        double joint_limit_stiffness,
                        double joint_limit_damping,
                        double joint_limit_max_torque) {
            auto params = makeJointImpedanceParams(
                stiffness,
                damping,
                constant_torque_offset,
                lower_joint_limits,
                upper_joint_limits,
                compensate_coriolis,
                max_delta_tau,
                joint_limit_activation_distance,
                joint_limit_stiffness,
                joint_limit_damping,
                joint_limit_max_torque);
            return std::make_shared<JointImpedanceTrackingMotion>(reference_handle, params);
          }),
          R"doc(Construct a dynamic joint impedance controller driven by a JointReferenceHandle.

Any constant_torque_offset configured here is added to the per-cycle torque_feedforward values published through the handle.)doc",
          "reference_handle"_a,
          "stiffness"_a = std::nullopt,
          "damping"_a = std::nullopt,
          "constant_torque_offset"_a = std::nullopt,
          "lower_joint_limits"_a = std::nullopt,
          "upper_joint_limits"_a = std::nullopt,
          "compensate_coriolis"_a = true,
          "max_delta_tau"_a = 1.0,
          "joint_limit_activation_distance"_a = 0.1,
          "joint_limit_stiffness"_a = 4.0,
          "joint_limit_damping"_a = 1.0,
          "joint_limit_max_torque"_a = 5.0)
      .def_property_readonly("target", &JointImpedanceTrackingMotion::target)
      .def_property_readonly("target_velocity", &JointImpedanceTrackingMotion::target_velocity);

  py::class_<ExponentialImpedanceMotion, CartesianImpedanceBase, std::shared_ptr<ExponentialImpedanceMotion>>(
      m, "ExponentialImpedanceMotion")
      .def(
          py::init<>([](const Affine &target,
                        ReferenceType target_type,
                        double translational_stiffness,
                        double rotational_stiffness,
                        std::optional<std::array<std::optional<double>, 6>>
                            force_constraints,
                        std::optional<Vector7d> nullspace_target,
                        double nullspace_stiffness,
                        double max_delta_tau,
                        std::optional<Vector7d> lower_joint_limits,
                        std::optional<Vector7d> upper_joint_limits,
                        double joint_limit_activation_distance,
                        double joint_limit_stiffness,
                        double joint_limit_damping,
                        double joint_limit_max_torque,
                        double exponential_decay = 0.005) {
            auto base_params = makeCartesianImpedanceParams(
                translational_stiffness,
                rotational_stiffness,
                force_constraints,
                nullspace_target,
                nullspace_stiffness,
                max_delta_tau,
                lower_joint_limits,
                upper_joint_limits,
                joint_limit_activation_distance,
                joint_limit_stiffness,
                joint_limit_damping,
                joint_limit_max_torque);
            auto params = ExponentialImpedanceMotion::Params{};
            static_cast<CartesianImpedanceBase::Params &>(params) = base_params;
            params.target_type = target_type;
            params.exponential_decay = exponential_decay;
            return std::make_shared<ExponentialImpedanceMotion>(
                target,
                params);
          }),
          R"doc(Construct an exponential Cartesian impedance motion toward a fixed target pose.

Cartesian damping is chosen internally as critically damped with respect to the requested stiffness.

The optional nullspace_target and nullspace_stiffness parameters add a secondary joint-posture objective that is projected into the Jacobian nullspace, so it biases the redundant arm posture without changing the Cartesian task to first order.)doc",
          "target"_a,
          py::arg_v("target_type", ReferenceType::kAbsolute, "_franky.ReferenceType.Absolute"),
          "translational_stiffness"_a = 2000,
          "rotational_stiffness"_a = 200,
          "force_constraints"_a = std::nullopt,
          "nullspace_target"_a = std::nullopt,
          "nullspace_stiffness"_a = 0.0,
          "max_delta_tau"_a = 1.0,
          "lower_joint_limits"_a = std::nullopt,
          "upper_joint_limits"_a = std::nullopt,
          "joint_limit_activation_distance"_a = 0.1,
          "joint_limit_stiffness"_a = 4.0,
          "joint_limit_damping"_a = 1.0,
          "joint_limit_max_torque"_a = 5.0,
          "exponential_decay"_a = 0.005);

  py::class_<CartesianImpedanceMotion, CartesianImpedanceBase, std::shared_ptr<CartesianImpedanceMotion>>(
      m, "CartesianImpedanceMotion")
      .def(
          py::init<>([](const Affine &target,
                        franka::Duration duration,
                        ReferenceType target_type,
                        double translational_stiffness,
                        double rotational_stiffness,
                        std::optional<std::array<std::optional<double>, 6>>
                            force_constraints,
                        std::optional<Vector7d> nullspace_target,
                        double nullspace_stiffness,
                        double max_delta_tau,
                        std::optional<Vector7d> lower_joint_limits,
                        std::optional<Vector7d> upper_joint_limits,
                        double joint_limit_activation_distance,
                        double joint_limit_stiffness,
                        double joint_limit_damping,
                        double joint_limit_max_torque,
                        bool return_when_finished,
                        double finish_wait_factor) {
            auto base_params = makeCartesianImpedanceParams(
                translational_stiffness,
                rotational_stiffness,
                force_constraints,
                nullspace_target,
                nullspace_stiffness,
                max_delta_tau,
                lower_joint_limits,
                upper_joint_limits,
                joint_limit_activation_distance,
                joint_limit_stiffness,
                joint_limit_damping,
                joint_limit_max_torque);
            auto params = CartesianImpedanceMotion::Params{};
            static_cast<CartesianImpedanceBase::Params &>(params) = base_params;
            params.target_type = target_type;
            params.return_when_finished = return_when_finished;
            params.finish_wait_factor = finish_wait_factor;
            return std::make_shared<CartesianImpedanceMotion>(
                target,
                duration,
                params);
          }),
          R"doc(Construct a Cartesian impedance motion that interpolates to a fixed target pose over the given duration.

Cartesian damping is chosen internally as critically damped with respect to the requested stiffness.

The optional nullspace_target and nullspace_stiffness parameters add a secondary joint-posture objective that is projected into the Jacobian nullspace, so it biases the redundant arm posture without changing the Cartesian task to first order.)doc",
          "target"_a,
          "duration"_a,
          py::arg_v("target_type", ReferenceType::kAbsolute, "_franky.ReferenceType.Absolute"),
          "translational_stiffness"_a = 2000,
          "rotational_stiffness"_a = 200,
          "force_constraints"_a = std::nullopt,
          "nullspace_target"_a = std::nullopt,
          "nullspace_stiffness"_a = 0.0,
          "max_delta_tau"_a = 1.0,
          "lower_joint_limits"_a = std::nullopt,
          "upper_joint_limits"_a = std::nullopt,
          "joint_limit_activation_distance"_a = 0.1,
          "joint_limit_stiffness"_a = 4.0,
          "joint_limit_damping"_a = 1.0,
          "joint_limit_max_torque"_a = 5.0,
          "return_when_finished"_a = true,
          "finish_wait_factor"_a = 1.2)
      ;

  py::class_<CartesianImpedanceTrackingMotion, CartesianImpedanceBase, std::shared_ptr<CartesianImpedanceTrackingMotion>>(
      m, "CartesianImpedanceTrackingMotion")
      .def(
          py::init<>([](const std::shared_ptr<CartesianReferenceHandle> &reference_handle,
                        double translational_stiffness,
                        double rotational_stiffness,
                        std::optional<std::array<std::optional<double>, 6>> force_constraints,
                        std::optional<Vector7d> nullspace_target,
                        double nullspace_stiffness,
                        double max_delta_tau,
                        std::optional<Vector7d> lower_joint_limits,
                        std::optional<Vector7d> upper_joint_limits,
                        double joint_limit_activation_distance,
                        double joint_limit_stiffness,
                        double joint_limit_damping,
                        double joint_limit_max_torque) {
            auto base_params = makeCartesianImpedanceParams(
                translational_stiffness,
                rotational_stiffness,
                force_constraints,
                nullspace_target,
                nullspace_stiffness,
                max_delta_tau,
                lower_joint_limits,
                upper_joint_limits,
                joint_limit_activation_distance,
                joint_limit_stiffness,
                joint_limit_damping,
                joint_limit_max_torque);
            return std::make_shared<CartesianImpedanceTrackingMotion>(
                reference_handle,
                base_params);
          }),
          R"doc(Construct a dynamic Cartesian impedance tracking controller driven by a CartesianReferenceHandle.

Each published Cartesian reference may optionally include a desired end-effector twist in the base frame.
Cartesian damping is chosen internally as critically damped with respect to the requested stiffness.
The optional nullspace_target and nullspace_stiffness parameters add a secondary joint-posture objective that is projected into the Jacobian nullspace, so it biases the redundant arm posture without changing the Cartesian task to first order.)doc",
          "reference_handle"_a,
          "translational_stiffness"_a = 2000,
          "rotational_stiffness"_a = 200,
          "force_constraints"_a = std::nullopt,
          "nullspace_target"_a = std::nullopt,
          "nullspace_stiffness"_a = 0.0,
          "max_delta_tau"_a = 1.0,
          "lower_joint_limits"_a = std::nullopt,
          "upper_joint_limits"_a = std::nullopt,
          "joint_limit_activation_distance"_a = 0.1,
          "joint_limit_stiffness"_a = 4.0,
          "joint_limit_damping"_a = 1.0,
          "joint_limit_max_torque"_a = 5.0);
}
