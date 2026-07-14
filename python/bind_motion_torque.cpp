#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "docstrings.hpp"
#include "franky.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the '_a' literal
using namespace franky;

namespace {

CartesianReference toCartesianReference(
    const Affine &target, const std::optional<Twist> &target_twist,
    const std::optional<TwistAcceleration> &target_acceleration) {
  CartesianReference reference;
  reference.target = target;
  reference.target_twist = target_twist;
  reference.target_acceleration = target_acceleration;
  return reference;
}

JointReference toJointReference(
    const Vector7d &position, const std::optional<Vector7d> &velocity,
    const std::optional<Vector7d> &torque_feedforward) {
  JointReference reference;
  reference.q = position;
  if (velocity.has_value()) reference.dq = velocity.value();
  if (torque_feedforward.has_value()) reference.tau_ff = torque_feedforward.value();
  return reference;
}

JointImpedanceParams makeJointImpedanceParams(
    const std::optional<Vector7d> &stiffness, const std::optional<Vector7d> &damping,
    const std::optional<Vector6d> &cartesian_stiffness, const std::optional<Vector6d> &cartesian_damping,
    const std::optional<Vector7d> &constant_torque_offset, const std::optional<Vector7d> &lower_joint_limits,
    const std::optional<Vector7d> &upper_joint_limits, bool compensate_coriolis, double max_delta_tau,
    double joint_limit_activation_distance, double joint_limit_stiffness, double joint_limit_damping,
    double joint_limit_max_torque, const std::optional<Vector7d> &friction_coulomb,
    const std::optional<Vector7d> &friction_viscous, const std::optional<Vector7d> &friction_max_torque,
    double friction_velocity_epsilon) {
  auto params = JointImpedanceParams{};
  if (stiffness.has_value()) {
    params.stiffness = stiffness.value();
    if (!damping.has_value()) params.damping = defaultJointImpedanceDamping(params.stiffness);
  }
  if (damping.has_value()) params.damping = damping.value();
  if (cartesian_stiffness.has_value()) {
    params.cartesian_gains = CartesianImpedanceGains::diagonal(*cartesian_stiffness, cartesian_damping);
  } else if (cartesian_damping.has_value()) {
    throw py::value_error("cartesian_damping requires cartesian_stiffness to be set");
  }
  if (friction_coulomb.has_value()) params.friction.coulomb = friction_coulomb.value();
  if (friction_viscous.has_value()) params.friction.viscous = friction_viscous.value();
  if (friction_max_torque.has_value()) params.friction.max_torque = friction_max_torque.value();
  if (constant_torque_offset.has_value()) params.constant_torque_offset = constant_torque_offset.value();
  params.safety.lower_joint_limits = lower_joint_limits;
  params.safety.upper_joint_limits = upper_joint_limits;
  params.compensate_coriolis = compensate_coriolis;
  params.friction.velocity_epsilon = friction_velocity_epsilon;
  params.safety.max_delta_tau = max_delta_tau;
  params.safety.joint_limit_activation_distance = joint_limit_activation_distance;
  params.safety.joint_limit_stiffness = joint_limit_stiffness;
  params.safety.joint_limit_damping = joint_limit_damping;
  params.safety.joint_limit_max_torque = joint_limit_max_torque;
  return params;
}

// Per-joint gains may be given as a scalar (applied to all joints) or a 7-vector.
using JointGain = std::variant<double, Vector7d>;

Vector7d broadcastJointGain(const JointGain &value) {
  if (std::holds_alternative<double>(value)) return Vector7d::Constant(std::get<double>(value));
  return std::get<Vector7d>(value);
}

CartesianImpedanceBase::Params makeCartesianImpedanceParams(
    double translational_stiffness, double rotational_stiffness,
    const std::optional<std::array<std::optional<double>, 6>> &force_constraints,
    const std::optional<Vector7d> &nullspace_target, const JointGain &nullspace_stiffness, double max_delta_tau,
    const std::optional<Vector7d> &lower_joint_limits, const std::optional<Vector7d> &upper_joint_limits,
    double joint_limit_activation_distance, double joint_limit_stiffness, double joint_limit_damping,
    double joint_limit_max_torque, const Eigen::Vector3d &translational_error_clip,
    const Eigen::Vector3d &rotational_error_clip, const std::optional<FrictionCompensationParams> &friction) {
  auto params = CartesianImpedanceBase::Params{};
  params.stiffness = cartesianGainBlocks(translational_stiffness, rotational_stiffness);
  params.translational_error_clip = translational_error_clip;
  params.rotational_error_clip = rotational_error_clip;
  if (nullspace_target.has_value()) {
    params.nullspace_tasks.emplace_back(PostureTask(*nullspace_target, broadcastJointGain(nullspace_stiffness)));
  }
  params.safety.max_delta_tau = max_delta_tau;
  params.safety.joint_limit_activation_distance = joint_limit_activation_distance;
  params.safety.joint_limit_stiffness = joint_limit_stiffness;
  params.safety.joint_limit_damping = joint_limit_damping;
  params.safety.joint_limit_max_torque = joint_limit_max_torque;
  params.safety.lower_joint_limits = lower_joint_limits;
  params.safety.upper_joint_limits = upper_joint_limits;
  if (force_constraints.has_value()) params.force_constraints = force_constraints.value();
  if (friction.has_value()) params.friction = *friction;
  return params;
}

}  // namespace

void bind_motion_torque(py::module &m) {
  py::class_<CartesianImpedanceGains>(m, "CartesianImpedanceGains", DOC(franky, CartesianImpedanceGains))
      .def(
          py::init<>([](double translational_stiffness, double rotational_stiffness) {
            return CartesianImpedanceGains::isotropic(translational_stiffness, rotational_stiffness);
          }),
          "translational_stiffness"_a = 500.0,
          "rotational_stiffness"_a = 50.0,
          "Construct isotropic gains with critical damping. Use the isotropic or diagonal factory functions for more "
          "control over the damping.")
      .def_static(
          "isotropic",
          &CartesianImpedanceGains::isotropic,
          "translational_stiffness"_a,
          "rotational_stiffness"_a,
          "translational_damping"_a = std::nullopt,
          "rotational_damping"_a = std::nullopt,
          DOC(franky, CartesianImpedanceGains, isotropic))
      .def_static(
          "diagonal",
          &CartesianImpedanceGains::diagonal,
          "stiffness"_a,
          "damping"_a = std::nullopt,
          DOC(franky, CartesianImpedanceGains, diagonal))
      .def_readwrite("stiffness", &CartesianImpedanceGains::stiffness, DOC(franky, CartesianImpedanceGains, stiffness))
      .def_readwrite("damping", &CartesianImpedanceGains::damping, DOC(franky, CartesianImpedanceGains, damping));

  py::class_<PostureTask>(m, "PostureTask", DOC(franky, PostureTask))
      .def(
          py::init<const Vector7d &, double, std::optional<double>, std::optional<double>>(),
          "target"_a,
          "stiffness"_a,
          "damping"_a = std::nullopt,
          "max_torque"_a = std::nullopt,
          DOC(franky, PostureTask, PostureTask_3))
      .def_readwrite("target", &PostureTask::target, DOC(franky, PostureTask, target))
      .def_readwrite("stiffness", &PostureTask::stiffness, DOC(franky, PostureTask, stiffness))
      .def_readwrite("damping", &PostureTask::damping, DOC(franky, PostureTask, damping))
      .def_readwrite("max_torque", &PostureTask::max_torque, DOC(franky, PostureTask, max_torque));

  py::class_<ManipulabilityTask>(m, "ManipulabilityTask", DOC(franky, ManipulabilityTask))
      .def(
          py::init<double, double, std::optional<double>>(),
          "gain"_a,
          "damping"_a = 0.0,
          "max_torque"_a = std::nullopt,
          DOC(franky, ManipulabilityTask, ManipulabilityTask_2))
      .def_readwrite("gain", &ManipulabilityTask::gain, DOC(franky, ManipulabilityTask, gain))
      .def_readwrite("damping", &ManipulabilityTask::damping, DOC(franky, ManipulabilityTask, damping))
      .def_readwrite("max_torque", &ManipulabilityTask::max_torque, DOC(franky, ManipulabilityTask, max_torque));

  py::class_<NullspaceGains>(m, "NullspaceGains", DOC(franky, NullspaceGains))
      .def(py::init<>())
      .def_property(
          "posture_stiffness",
          [](const NullspaceGains &gains) { return gains.posture_stiffness; },
          [](NullspaceGains &gains, const JointGain &value) { gains.posture_stiffness = broadcastJointGain(value); },
          "Per-joint posture stiffness [Nm/rad]. Accepts a scalar (applied to all joints) or a 7-vector.")
      .def_property(
          "posture_damping",
          [](const NullspaceGains &gains) { return gains.posture_damping; },
          [](NullspaceGains &gains, const std::optional<JointGain> &value) {
            if (value.has_value()) {
              gains.posture_damping = broadcastJointGain(*value);
            } else {
              gains.posture_damping = std::nullopt;
            }
          },
          "Per-joint posture damping [Nms/rad]. Accepts a scalar (applied to all joints), a 7-vector, or None for "
          "critical damping.")
      .def_readwrite(
          "posture_max_torque", &NullspaceGains::posture_max_torque, DOC(franky, NullspaceGains, posture_max_torque))
      .def_readwrite(
          "manipulability_gain", &NullspaceGains::manipulability_gain, DOC(franky, NullspaceGains, manipulability_gain))
      .def_readwrite(
          "manipulability_damping",
          &NullspaceGains::manipulability_damping,
          DOC(franky, NullspaceGains, manipulability_damping))
      .def_readwrite(
          "manipulability_max_torque",
          &NullspaceGains::manipulability_max_torque,
          DOC(franky, NullspaceGains, manipulability_max_torque));

  py::class_<CartesianReference>(m, "CartesianReference", DOC(franky, CartesianReference))
      .def(
          py::init<>(
              [](const Affine &target,
                 std::optional<Twist>
                     target_twist,
                 std::optional<TwistAcceleration>
                     target_acceleration) { return CartesianReference{target, target_twist, target_acceleration}; }),
          "target"_a = Affine::Identity(),
          "target_twist"_a = std::nullopt,
          "target_acceleration"_a = std::nullopt)
      .def_readwrite("target", &CartesianReference::target, DOC(franky, CartesianReference, target))
      .def_readwrite("target_twist", &CartesianReference::target_twist, DOC(franky, CartesianReference, target_twist))
      .def_readwrite(
          "target_acceleration",
          &CartesianReference::target_acceleration,
          DOC(franky, CartesianReference, target_acceleration));

  py::class_<JointImpedanceGains>(m, "JointImpedanceGains", DOC(franky, JointImpedanceGains))
      .def(
          py::init<const std::optional<Vector7d> &, const std::optional<Vector7d> &>(),
          "stiffness"_a = std::nullopt,
          "damping"_a = std::nullopt,
          "Construct joint impedance gains. If stiffness is unset, the default stiffness is used. If damping is "
          "unset, critical damping is used.")
      .def_readwrite("stiffness", &JointImpedanceGains::stiffness, DOC(franky, JointImpedanceGains, stiffness))
      .def_readwrite("damping", &JointImpedanceGains::damping, DOC(franky, JointImpedanceGains, damping));

  py::class_<JointReference>(m, "JointReference", DOC(franky, JointReference))
      .def(
          py::init<>([](std::optional<Vector7d> q, std::optional<Vector7d> dq, std::optional<Vector7d> tau_ff) {
            JointReference ref;
            if (q) ref.q = *q;
            if (dq) ref.dq = *dq;
            if (tau_ff) ref.tau_ff = *tau_ff;
            return ref;
          }),
          "q"_a = std::nullopt,
          "dq"_a = std::nullopt,
          "tau_ff"_a = std::nullopt,
          "Construct a joint reference. Unset components default to zero.")
      .def_readwrite("q", &JointReference::q, DOC(franky, JointReference, q))
      .def_readwrite("dq", &JointReference::dq, DOC(franky, JointReference, dq))
      .def_readwrite("tau_ff", &JointReference::tau_ff, DOC(franky, JointReference, tau_ff));

  auto cartesian_impedance_base =
      py::class_<CartesianImpedanceBase, Motion<franka::Torques>, std::shared_ptr<CartesianImpedanceBase>>(
          m, "CartesianImpedanceBase", DOC(franky, CartesianImpedanceBase))
          .def("get_gains", &CartesianImpedanceBase::getGains, DOC(franky, CartesianImpedanceBase, getGains))
          .def("set_gains", &CartesianImpedanceBase::setGains, "gains"_a, DOC(franky, CartesianImpedanceBase, setGains))
          .def(
              "get_nullspace_gains",
              &CartesianImpedanceBase::getNullspaceGains,
              DOC(franky, CartesianImpedanceBase, getNullspaceGains))
          .def(
              "set_nullspace_gains",
              &CartesianImpedanceBase::setNullspaceGains,
              "gains"_a,
              DOC(franky, CartesianImpedanceBase, setNullspaceGains));
  m.attr("ImpedanceMotion") = cartesian_impedance_base;

  py::class_<JointImpedanceBase, Motion<franka::Torques>, std::shared_ptr<JointImpedanceBase>>(
      m, "JointImpedanceBase", DOC(franky, JointImpedanceBase))
      .def("get_gains", &JointImpedanceBase::getGains, DOC(franky, JointImpedanceBase, getGains))
      .def("set_gains", &JointImpedanceBase::setGains, "gains"_a, DOC(franky, JointImpedanceBase, setGains))
      .def(
          "get_cartesian_gains",
          &JointImpedanceBase::getCartesianGains,
          DOC(franky, JointImpedanceBase, getCartesianGains))
      .def(
          "set_cartesian_gains",
          &JointImpedanceBase::setCartesianGains,
          "gains"_a,
          DOC(franky, JointImpedanceBase, setCartesianGains));

  // Params classes — bind before the motion classes that use them.

  py::class_<TorqueSafetyParams>(m, "TorqueSafetyParams", DOC(franky, TorqueSafetyParams))
      .def(py::init<>())
      .def_readwrite(
          "max_delta_tau", &TorqueSafetyParams::max_delta_tau, DOC(franky, TorqueSafetyParams, max_delta_tau))
      .def_readwrite(
          "lower_joint_limits",
          &TorqueSafetyParams::lower_joint_limits,
          DOC(franky, TorqueSafetyParams, lower_joint_limits))
      .def_readwrite(
          "upper_joint_limits",
          &TorqueSafetyParams::upper_joint_limits,
          DOC(franky, TorqueSafetyParams, upper_joint_limits))
      .def_readwrite(
          "joint_limit_activation_distance",
          &TorqueSafetyParams::joint_limit_activation_distance,
          DOC(franky, TorqueSafetyParams, joint_limit_activation_distance))
      .def_readwrite(
          "joint_limit_stiffness",
          &TorqueSafetyParams::joint_limit_stiffness,
          DOC(franky, TorqueSafetyParams, joint_limit_stiffness))
      .def_readwrite(
          "joint_limit_damping",
          &TorqueSafetyParams::joint_limit_damping,
          DOC(franky, TorqueSafetyParams, joint_limit_damping))
      .def_readwrite(
          "joint_limit_max_torque",
          &TorqueSafetyParams::joint_limit_max_torque,
          DOC(franky, TorqueSafetyParams, joint_limit_max_torque));

  py::class_<FrictionCompensationParams>(m, "FrictionCompensationParams", DOC(franky, FrictionCompensationParams))
      .def(
          py::init<>([](const std::array<double, 7> &coulomb,
                        const std::array<double, 7> &viscous,
                        const std::array<double, 7> &max_torque,
                        double velocity_epsilon) {
            return FrictionCompensationParams(
                toEigenD<7>(coulomb), toEigenD<7>(viscous), toEigenD<7>(max_torque), velocity_epsilon);
          }),
          "coulomb"_a = std::array<double, 7>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
          "viscous"_a = std::array<double, 7>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
          "max_torque"_a = std::array<double, 7>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
          "velocity_epsilon"_a = 0.03)
      .def_readwrite("coulomb", &FrictionCompensationParams::coulomb, DOC(franky, FrictionCompensationParams, coulomb))
      .def_readwrite("viscous", &FrictionCompensationParams::viscous, DOC(franky, FrictionCompensationParams, viscous))
      .def_readwrite(
          "max_torque", &FrictionCompensationParams::max_torque, DOC(franky, FrictionCompensationParams, max_torque))
      .def_readwrite(
          "velocity_epsilon",
          &FrictionCompensationParams::velocity_epsilon,
          DOC(franky, FrictionCompensationParams, velocity_epsilon));

  py::class_<JointImpedanceParams>(m, "JointImpedanceParams", DOC(franky, JointImpedanceParams))
      .def(py::init<>())
      .def_readwrite("stiffness", &JointImpedanceParams::stiffness, DOC(franky, JointImpedanceParams, stiffness))
      .def_readwrite("damping", &JointImpedanceParams::damping, DOC(franky, JointImpedanceParams, damping))
      .def_readwrite("error_clip", &JointImpedanceParams::error_clip, DOC(franky, JointImpedanceParams, error_clip))
      .def_readwrite(
          "constant_torque_offset",
          &JointImpedanceParams::constant_torque_offset,
          DOC(franky, JointImpedanceParams, constant_torque_offset))
      .def_readwrite(
          "compensate_coriolis",
          &JointImpedanceParams::compensate_coriolis,
          DOC(franky, JointImpedanceParams, compensate_coriolis))
      .def_readwrite(
          "cartesian_gains", &JointImpedanceParams::cartesian_gains, DOC(franky, JointImpedanceParams, cartesian_gains))
      .def_readwrite("safety", &JointImpedanceParams::safety, DOC(franky, JointImpedanceParams, safety))
      .def_readwrite("friction", &JointImpedanceParams::friction, DOC(franky, JointImpedanceParams, friction));

  py::class_<CartesianImpedanceBase::Params>(m, "CartesianImpedanceParams", DOC(franky, CartesianImpedanceBase, Params))
      .def(py::init<>())
      .def_readwrite(
          "stiffness",
          &CartesianImpedanceBase::Params::stiffness,
          DOC(franky, CartesianImpedanceBase, Params, stiffness))
      .def_readwrite(
          "damping", &CartesianImpedanceBase::Params::damping, DOC(franky, CartesianImpedanceBase, Params, damping))
      .def_readwrite(
          "translational_error_clip",
          &CartesianImpedanceBase::Params::translational_error_clip,
          DOC(franky, CartesianImpedanceBase, Params, translational_error_clip))
      .def_readwrite(
          "rotational_error_clip",
          &CartesianImpedanceBase::Params::rotational_error_clip,
          DOC(franky, CartesianImpedanceBase, Params, rotational_error_clip))
      .def_readwrite(
          "force_constraints",
          &CartesianImpedanceBase::Params::force_constraints,
          DOC(franky, CartesianImpedanceBase, Params, force_constraints))
      .def_readwrite(
          "nullspace_tasks",
          &CartesianImpedanceBase::Params::nullspace_tasks,
          DOC(franky, CartesianImpedanceBase, Params, nullspace_tasks))
      .def_readwrite(
          "safety", &CartesianImpedanceBase::Params::safety, DOC(franky, CartesianImpedanceBase, Params, safety))
      .def_readwrite(
          "friction", &CartesianImpedanceBase::Params::friction, DOC(franky, CartesianImpedanceBase, Params, friction));

  py::class_<CartesianImpedanceMotion::Params, CartesianImpedanceBase::Params>(
      m, "CartesianImpedanceMotionParams", DOC(franky, CartesianImpedanceMotion, Params))
      .def(py::init<>())
      .def_readwrite(
          "target_type",
          &CartesianImpedanceMotion::Params::target_type,
          DOC(franky, CartesianImpedanceMotion, Params, target_type));

  py::class_<JointImpedanceMotion, JointImpedanceBase, std::shared_ptr<JointImpedanceMotion>>(
      m, "JointImpedanceMotion", DOC(franky, JointImpedanceMotion))
      .def(
          py::init<>([](const Vector7d &target,
                        std::optional<Vector7d>
                            target_velocity,
                        std::optional<Vector7d>
                            stiffness,
                        std::optional<Vector7d>
                            damping,
                        std::optional<Vector6d>
                            cartesian_stiffness,
                        std::optional<Vector6d>
                            cartesian_damping,
                        std::optional<Vector7d>
                            constant_torque_offset,
                        std::optional<Vector7d>
                            lower_joint_limits,
                        std::optional<Vector7d>
                            upper_joint_limits,
                        bool compensate_coriolis,
                        double max_delta_tau,
                        double joint_limit_activation_distance,
                        double joint_limit_stiffness,
                        double joint_limit_damping,
                        double joint_limit_max_torque,
                        std::optional<Vector7d>
                            friction_coulomb,
                        std::optional<Vector7d>
                            friction_viscous,
                        std::optional<Vector7d>
                            friction_max_torque,
                        double friction_velocity_epsilon) {
            auto params = makeJointImpedanceParams(
                stiffness,
                damping,
                cartesian_stiffness,
                cartesian_damping,
                constant_torque_offset,
                lower_joint_limits,
                upper_joint_limits,
                compensate_coriolis,
                max_delta_tau,
                joint_limit_activation_distance,
                joint_limit_stiffness,
                joint_limit_damping,
                joint_limit_max_torque,
                friction_coulomb,
                friction_viscous,
                friction_max_torque,
                friction_velocity_epsilon);

            const Vector7d target_vector = target;
            if (target_velocity.has_value()) {
              return std::make_shared<JointImpedanceMotion>(target_vector, target_velocity.value(), params);
            }
            return std::make_shared<JointImpedanceMotion>(target_vector, params);
          }),
          R"doc(Construct a joint impedance controller regulating toward a fixed joint target.

This motion does not finish on its own; it keeps regulating toward the target until it is
preempted or stopped.

If damping is not set, critical damping is chosen with respect to the given stiffness. The
optional cartesian_stiffness/cartesian_damping parameters add a Cartesian-space spring-damper
term that is projected through the current Jacobian. The joint-limit parameters configure a
soft repulsion torque that activates near the joint limits (only if both lower_joint_limits
and upper_joint_limits are set).)doc",
          "target"_a,
          "target_velocity"_a = std::nullopt,
          "stiffness"_a = std::nullopt,
          "damping"_a = std::nullopt,
          "cartesian_stiffness"_a = std::nullopt,
          "cartesian_damping"_a = std::nullopt,
          "constant_torque_offset"_a = std::nullopt,
          "lower_joint_limits"_a = std::nullopt,
          "upper_joint_limits"_a = std::nullopt,
          "compensate_coriolis"_a = true,
          "max_delta_tau"_a = 1.0,
          "joint_limit_activation_distance"_a = 0.1,
          "joint_limit_stiffness"_a = 4.0,
          "joint_limit_damping"_a = 1.0,
          "joint_limit_max_torque"_a = 5.0,
          "friction_coulomb"_a = std::nullopt,
          "friction_viscous"_a = std::nullopt,
          "friction_max_torque"_a = std::nullopt,
          "friction_velocity_epsilon"_a = 0.03)
      .def_property_readonly("target", &JointImpedanceMotion::target, DOC(franky, JointImpedanceBase, target))
      .def_property_readonly(
          "target_velocity", &JointImpedanceMotion::target_velocity, DOC(franky, JointImpedanceBase, target_velocity))
      .def_property_readonly(
          "params", [](const JointImpedanceMotion &m) { return m.params(); }, DOC(franky, JointImpedanceBase, params));

  py::class_<JointImpedanceTrackingMotion, JointImpedanceBase, std::shared_ptr<JointImpedanceTrackingMotion>>(
      m, "JointImpedanceTrackingMotion", DOC(franky, JointImpedanceTrackingMotion))
      .def(
          py::init<>([](std::optional<Vector7d> stiffness,
                        std::optional<Vector7d>
                            damping,
                        std::optional<Vector6d>
                            cartesian_stiffness,
                        std::optional<Vector6d>
                            cartesian_damping,
                        std::optional<Vector7d>
                            constant_torque_offset,
                        std::optional<Vector7d>
                            lower_joint_limits,
                        std::optional<Vector7d>
                            upper_joint_limits,
                        bool compensate_coriolis,
                        double max_delta_tau,
                        double joint_limit_activation_distance,
                        double joint_limit_stiffness,
                        double joint_limit_damping,
                        double joint_limit_max_torque,
                        std::optional<Vector7d>
                            friction_coulomb,
                        std::optional<Vector7d>
                            friction_viscous,
                        std::optional<Vector7d>
                            friction_max_torque,
                        double friction_velocity_epsilon,
                        double gains_time_constant) {
            auto params = makeJointImpedanceParams(
                stiffness,
                damping,
                cartesian_stiffness,
                cartesian_damping,
                constant_torque_offset,
                lower_joint_limits,
                upper_joint_limits,
                compensate_coriolis,
                max_delta_tau,
                joint_limit_activation_distance,
                joint_limit_stiffness,
                joint_limit_damping,
                joint_limit_max_torque,
                friction_coulomb,
                friction_viscous,
                friction_max_torque,
                friction_velocity_epsilon);
            return std::make_shared<JointImpedanceTrackingMotion>(params, gains_time_constant);
          }),
          R"doc(Construct a dynamic joint impedance controller.

Any constant_torque_offset configured here is added to the per-cycle torque_feedforward values.

The controller reads target gains each cycle and exponentially
interpolates toward them with the given time constant, allowing smooth runtime stiffness/damping changes.)doc",
          "stiffness"_a = std::nullopt,
          "damping"_a = std::nullopt,
          "cartesian_stiffness"_a = std::nullopt,
          "cartesian_damping"_a = std::nullopt,
          "constant_torque_offset"_a = std::nullopt,
          "lower_joint_limits"_a = std::nullopt,
          "upper_joint_limits"_a = std::nullopt,
          "compensate_coriolis"_a = true,
          "max_delta_tau"_a = 1.0,
          "joint_limit_activation_distance"_a = 0.1,
          "joint_limit_stiffness"_a = 4.0,
          "joint_limit_damping"_a = 1.0,
          "joint_limit_max_torque"_a = 5.0,
          "friction_coulomb"_a = std::nullopt,
          "friction_viscous"_a = std::nullopt,
          "friction_max_torque"_a = std::nullopt,
          "friction_velocity_epsilon"_a = 0.03,
          "gains_time_constant"_a = 0.1)
      .def_property_readonly("target", &JointImpedanceTrackingMotion::target, DOC(franky, JointImpedanceBase, target))
      .def_property_readonly(
          "target_velocity",
          &JointImpedanceTrackingMotion::target_velocity,
          DOC(franky, JointImpedanceBase, target_velocity))
      .def_property_readonly(
          "params",
          [](const JointImpedanceTrackingMotion &m) { return m.params(); },
          DOC(franky, JointImpedanceBase, params))
      .def(
          "get_reference",
          [](const JointImpedanceTrackingMotion &m) { return m.getReference(); },
          "Get a copy of the last published joint reference, or None if no reference has been set yet. Mutating the "
          "returned object has no effect on the motion; pass it to set_reference to apply changes.")
      .def(
          "set_reference",
          &JointImpedanceTrackingMotion::setReference,
          "reference"_a,
          "Set the joint reference tracked by the controller. The reference is validated and picked up by the "
          "control loop in the next cycle.");

  py::class_<CartesianImpedanceMotion, CartesianImpedanceBase, std::shared_ptr<CartesianImpedanceMotion>>(
      m, "CartesianImpedanceMotion", DOC(franky, CartesianImpedanceMotion))
      .def(
          py::init<>([](const Affine &target,
                        std::optional<Twist>
                            target_twist,
                        ReferenceType target_type,
                        double translational_stiffness,
                        double rotational_stiffness,
                        std::optional<std::array<std::optional<double>, 6>>
                            force_constraints,
                        std::optional<Vector7d>
                            nullspace_target,
                        const JointGain &nullspace_stiffness,
                        double max_delta_tau,
                        std::optional<Vector7d>
                            lower_joint_limits,
                        std::optional<Vector7d>
                            upper_joint_limits,
                        double joint_limit_activation_distance,
                        double joint_limit_stiffness,
                        double joint_limit_damping,
                        double joint_limit_max_torque,
                        const Eigen::Vector3d &translational_error_clip,
                        const Eigen::Vector3d &rotational_error_clip,
                        std::optional<FrictionCompensationParams>
                            friction) {
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
                joint_limit_max_torque,
                translational_error_clip,
                rotational_error_clip,
                friction);
            auto params = CartesianImpedanceMotion::Params{};
            static_cast<CartesianImpedanceBase::Params &>(params) = base_params;
            params.target_type = target_type;
            if (target_twist.has_value()) {
              return std::make_shared<CartesianImpedanceMotion>(target, target_twist.value(), params);
            }
            return std::make_shared<CartesianImpedanceMotion>(target, params);
          }),
          R"doc(Construct a Cartesian impedance motion that regulates toward a fixed target pose.

Like JointImpedanceMotion, this motion does not finish on its own; it keeps regulating toward the target until it is preempted or stopped.

If target_twist is provided, it is interpreted as the desired end-effector twist in the base frame and the damping term acts on twist error rather than damping all motion toward zero.

Cartesian damping is chosen internally as critically damped with respect to the requested stiffness.

The optional nullspace_target and nullspace_stiffness parameters add a secondary joint-posture objective that is projected into the Jacobian nullspace, so it biases the redundant arm posture without changing the Cartesian task to first order. nullspace_stiffness accepts a scalar (applied to all joints) or a per-joint 7-vector; joints with zero stiffness are not pushed by the posture objective.)doc",
          "target"_a,
          "target_twist"_a = std::nullopt,
          py::arg_v("target_type", ReferenceType::kAbsolute, "_franky.ReferenceType.Absolute"),
          "translational_stiffness"_a = 500,
          "rotational_stiffness"_a = 50,
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
          "translational_error_clip"_a = Eigen::Vector3d::Constant(0.10),
          "rotational_error_clip"_a = Eigen::Vector3d::Constant(0.25),
          "friction"_a = std::nullopt)
      .def_property_readonly("target", &CartesianImpedanceMotion::target, DOC(franky, CartesianImpedanceBase, target))
      .def_property_readonly(
          "target_twist", &CartesianImpedanceMotion::target_twist, DOC(franky, CartesianImpedanceMotion, target_twist))
      .def_property_readonly(
          "params",
          [](const CartesianImpedanceMotion &m) { return m.params(); },
          DOC(franky, CartesianImpedanceMotion, params));

  py::class_<
      CartesianImpedanceTrackingMotion,
      CartesianImpedanceBase,
      std::shared_ptr<CartesianImpedanceTrackingMotion>>(
      m, "CartesianImpedanceTrackingMotion", DOC(franky, CartesianImpedanceTrackingMotion))
      .def(
          py::init<>([](double translational_stiffness,
                        double rotational_stiffness,
                        std::optional<std::array<std::optional<double>, 6>>
                            force_constraints,
                        std::optional<Vector7d>
                            nullspace_target,
                        const JointGain &nullspace_stiffness,
                        double max_delta_tau,
                        std::optional<Vector7d>
                            lower_joint_limits,
                        std::optional<Vector7d>
                            upper_joint_limits,
                        double joint_limit_activation_distance,
                        double joint_limit_stiffness,
                        double joint_limit_damping,
                        double joint_limit_max_torque,
                        const Eigen::Vector3d &translational_error_clip,
                        const Eigen::Vector3d &rotational_error_clip,
                        double gains_time_constant,
                        std::optional<FrictionCompensationParams>
                            friction) {
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
                joint_limit_max_torque,
                translational_error_clip,
                rotational_error_clip,
                friction);
            return std::make_shared<CartesianImpedanceTrackingMotion>(base_params, gains_time_constant);
          }),
          R"doc(Construct a dynamic Cartesian impedance tracking controller.

Cartesian damping is chosen internally as critically damped with respect to the requested stiffness.
The optional nullspace_target and nullspace_stiffness parameters add a secondary joint-posture objective that is projected into the Jacobian nullspace, so it biases the redundant arm posture without changing the Cartesian task to first order. nullspace_stiffness accepts a scalar (applied to all joints) or a per-joint 7-vector; joints with zero stiffness are not pushed by the posture objective.

The controller reads target gains each cycle and exponentially
interpolates toward them with the given time constant, allowing smooth runtime stiffness changes.)doc",
          "translational_stiffness"_a = 500,
          "rotational_stiffness"_a = 50,
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
          "translational_error_clip"_a = Eigen::Vector3d::Constant(0.10),
          "rotational_error_clip"_a = Eigen::Vector3d::Constant(0.25),
          "gains_time_constant"_a = 0.1,
          "friction"_a = std::nullopt)
      .def_property_readonly(
          "params",
          [](const CartesianImpedanceTrackingMotion &m) { return m.params(); },
          DOC(franky, CartesianImpedanceTrackingMotion, params))
      .def(
          "get_reference",
          [](const CartesianImpedanceTrackingMotion &m) { return m.getReference(); },
          "Get a copy of the last published Cartesian reference, or None if no reference has been set yet. Mutating "
          "the returned object has no effect on the motion; pass it to set_reference to apply changes.")
      .def(
          "set_reference",
          &CartesianImpedanceTrackingMotion::setReference,
          "reference"_a,
          "Set the Cartesian reference tracked by the controller. The reference is validated and picked up by the "
          "control loop in the next cycle.");

  py::class_<TorqueStopParams>(m, "TorqueStopParams", DOC(franky, TorqueStopParams))
      .def(py::init<>())
      .def_readwrite("damping", &TorqueStopParams::damping, DOC(franky, TorqueStopParams, damping))
      .def_readwrite("ramp_duration", &TorqueStopParams::ramp_duration, DOC(franky, TorqueStopParams, ramp_duration))
      .def_readwrite(
          "velocity_epsilon", &TorqueStopParams::velocity_epsilon, DOC(franky, TorqueStopParams, velocity_epsilon))
      .def_readwrite("max_duration", &TorqueStopParams::max_duration, DOC(franky, TorqueStopParams, max_duration))
      .def_readwrite(
          "compensate_coriolis",
          &TorqueStopParams::compensate_coriolis,
          DOC(franky, TorqueStopParams, compensate_coriolis))
      .def_readwrite("max_delta_tau", &TorqueStopParams::max_delta_tau, DOC(franky, TorqueStopParams, max_delta_tau));

  py::class_<StopMotion<franka::Torques>, Motion<franka::Torques>, std::shared_ptr<StopMotion<franka::Torques>>>(
      m, "TorqueStopMotion", DOC(franky, StopMotion_6))
      .def(
          py::init<>([](std::optional<Vector7d> damping,
                        double ramp_duration,
                        double velocity_epsilon,
                        double max_duration,
                        bool compensate_coriolis,
                        double max_delta_tau) {
            auto params = TorqueStopParams{};
            if (damping.has_value()) {
              validateNonNegativeFinite(damping.value(), "damping");
              params.damping = damping.value();
            }
            validateNonNegativeFinite(ramp_duration, "ramp_duration");
            validateNonNegativeFinite(velocity_epsilon, "velocity_epsilon");
            validateNonNegativeFinite(max_duration, "max_duration");
            validateNonNegativeFinite(max_delta_tau, "max_delta_tau");
            params.ramp_duration = ramp_duration;
            params.velocity_epsilon = velocity_epsilon;
            params.max_duration = max_duration;
            params.compensate_coriolis = compensate_coriolis;
            params.max_delta_tau = max_delta_tau;
            return std::make_shared<StopMotion<franka::Torques>>(params);
          }),
          R"doc(Graceful stop for torque-control (impedance) motions.

Torque motions never signal MotionFinished on their own, and Robot.stop() preempts the
control loop with a ControlException rather than ramping down. Enqueue this motion (or
return it from a TorqueReaction) to end a torque loop cleanly: it blends the last
commanded torque into a zero-stiffness joint-damping law, brings the arm to rest, then
finishes. The motion also finishes after max_duration regardless of velocity, so a
sustained external push cannot make it hang.)doc",
          "damping"_a = std::nullopt,
          "ramp_duration"_a = 0.2,
          "velocity_epsilon"_a = 0.02,
          "max_duration"_a = 2.0,
          "compensate_coriolis"_a = true,
          "max_delta_tau"_a = 1.0);
}
