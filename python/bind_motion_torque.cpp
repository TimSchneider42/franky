#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

CartesianImpedanceBase::Params makeCartesianImpedanceParams(
    double translational_stiffness, double rotational_stiffness,
    const std::optional<std::array<std::optional<double>, 6>> &force_constraints,
    const std::optional<Vector7d> &nullspace_target, double nullspace_stiffness, double max_delta_tau,
    const std::optional<Vector7d> &lower_joint_limits, const std::optional<Vector7d> &upper_joint_limits,
    double joint_limit_activation_distance, double joint_limit_stiffness, double joint_limit_damping,
    double joint_limit_max_torque, const Eigen::Vector3d &translational_error_clip,
    const Eigen::Vector3d &rotational_error_clip, const std::optional<FrictionCompensationParams> &friction) {
  auto params = CartesianImpedanceBase::Params{};
  params.translational_stiffness = translational_stiffness;
  params.rotational_stiffness = rotational_stiffness;
  params.translational_error_clip = translational_error_clip;
  params.rotational_error_clip = rotational_error_clip;
  params.nullspace_target = nullspace_target;
  params.nullspace_stiffness = nullspace_stiffness;
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
  auto cartesian_impedance_base =
      py::class_<CartesianImpedanceBase, Motion<franka::Torques>, std::shared_ptr<CartesianImpedanceBase>>(
          m, "CartesianImpedanceBase");
  m.attr("ImpedanceMotion") = cartesian_impedance_base;

  py::class_<CartesianImpedanceGains>(m, "CartesianImpedanceGains")
      .def(
          py::init<double, double, double>(),
          "translational_stiffness"_a = 500.0,
          "rotational_stiffness"_a = 50.0,
          "nullspace_stiffness"_a = 0.0)
      .def_readwrite("translational_stiffness", &CartesianImpedanceGains::translational_stiffness)
      .def_readwrite("rotational_stiffness", &CartesianImpedanceGains::rotational_stiffness)
      .def_readwrite("nullspace_stiffness", &CartesianImpedanceGains::nullspace_stiffness);

  py::class_<CartesianImpedanceGainsHandle, std::shared_ptr<CartesianImpedanceGainsHandle>>(
      m, "CartesianImpedanceGainsHandle")
      .def(py::init<>())
      .def(
          "set",
          [](CartesianImpedanceGainsHandle &handle,
             double translational_stiffness,
             double rotational_stiffness,
             double nullspace_stiffness) {
            handle.set(CartesianImpedanceGains(translational_stiffness, rotational_stiffness, nullspace_stiffness));
          },
          "translational_stiffness"_a,
          "rotational_stiffness"_a,
          "nullspace_stiffness"_a = 0.0)
      .def("clear", &CartesianImpedanceGainsHandle::clear)
      .def("get", &CartesianImpedanceGainsHandle::get)
      .def_property_readonly("has_value", &CartesianImpedanceGainsHandle::hasValue);

  py::class_<JointImpedanceGains>(m, "JointImpedanceGains")
      .def(
          py::init<const std::optional<Vector7d> &, const std::optional<Vector7d> &>(),
          "stiffness"_a = std::nullopt,
          "damping"_a = std::nullopt)
      .def_readwrite("stiffness", &JointImpedanceGains::stiffness)
      .def_readwrite("damping", &JointImpedanceGains::damping);

  py::class_<JointImpedanceGainsHandle, std::shared_ptr<JointImpedanceGainsHandle>>(m, "JointImpedanceGainsHandle")
      .def(py::init<>())
      .def(
          "set",
          [](JointImpedanceGainsHandle &handle, const Vector7d &stiffness, const Vector7d &damping) {
            handle.set(JointImpedanceGains(stiffness, damping));
          },
          "stiffness"_a,
          "damping"_a)
      .def("clear", &JointImpedanceGainsHandle::clear)
      .def("get", &JointImpedanceGainsHandle::get)
      .def_property_readonly("has_value", &JointImpedanceGainsHandle::hasValue);

  py::class_<JointReferenceHandle, std::shared_ptr<JointReferenceHandle>>(m, "JointReferenceHandle")
      .def(py::init<>())
      .def(
          "set",
          [](JointReferenceHandle &handle,
             const Vector7d &position,
             std::optional<Vector7d>
                 velocity,
             std::optional<Vector7d>
                 torque_feedforward) { handle.set(toJointReference(position, velocity, torque_feedforward)); },
          R"doc(Update the current joint reference for a running JointImpedanceTrackingMotion.

The provided torque_feedforward is added on top of any constant_torque_offset configured on the motion.)doc",
          "position"_a,
          "velocity"_a = std::nullopt,
          "torque_feedforward"_a = std::nullopt)
      .def("clear", &JointReferenceHandle::clear)
      .def_property_readonly("has_value", &JointReferenceHandle::hasValue);

  py::class_<CartesianReferenceHandle, std::shared_ptr<CartesianReferenceHandle>>(m, "CartesianReferenceHandle")
      .def(py::init<>())
      .def(
          "set",
          [](CartesianReferenceHandle &handle,
             const Affine &target,
             std::optional<Twist>
                 target_twist,
             std::optional<TwistAcceleration>
                 target_acceleration) { handle.set(toCartesianReference(target, target_twist, target_acceleration)); },
          R"doc(Update the current Cartesian reference for a running CartesianImpedanceTrackingMotion.

If target_twist is provided, it is interpreted as the desired end-effector twist in the base frame and the damping term acts on twist error rather than motion relative to zero.

If target_acceleration is provided, it is interpreted as the desired end-effector acceleration in the base frame and contributes a model-based inertial feedforward wrench.)doc",
          "target"_a,
          "target_twist"_a = std::nullopt,
          "target_acceleration"_a = std::nullopt)
      .def("clear", &CartesianReferenceHandle::clear)
      .def_property_readonly("has_value", &CartesianReferenceHandle::hasValue);

  // Params classes — bind before the motion classes that use them.

  py::class_<TorqueSafetyParams>(m, "TorqueSafetyParams")
      .def(py::init<>())
      .def_readwrite("max_delta_tau", &TorqueSafetyParams::max_delta_tau)
      .def_readwrite("lower_joint_limits", &TorqueSafetyParams::lower_joint_limits)
      .def_readwrite("upper_joint_limits", &TorqueSafetyParams::upper_joint_limits)
      .def_readwrite("joint_limit_activation_distance", &TorqueSafetyParams::joint_limit_activation_distance)
      .def_readwrite("joint_limit_stiffness", &TorqueSafetyParams::joint_limit_stiffness)
      .def_readwrite("joint_limit_damping", &TorqueSafetyParams::joint_limit_damping)
      .def_readwrite("joint_limit_max_torque", &TorqueSafetyParams::joint_limit_max_torque);

  py::class_<FrictionCompensationParams>(m, "FrictionCompensationParams")
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
      .def_readwrite("coulomb", &FrictionCompensationParams::coulomb)
      .def_readwrite("viscous", &FrictionCompensationParams::viscous)
      .def_readwrite("max_torque", &FrictionCompensationParams::max_torque)
      .def_readwrite("velocity_epsilon", &FrictionCompensationParams::velocity_epsilon);

  py::class_<JointImpedanceParams>(m, "JointImpedanceParams")
      .def(py::init<>())
      .def_readwrite("stiffness", &JointImpedanceParams::stiffness)
      .def_readwrite("damping", &JointImpedanceParams::damping)
      .def_readwrite("constant_torque_offset", &JointImpedanceParams::constant_torque_offset)
      .def_readwrite("compensate_coriolis", &JointImpedanceParams::compensate_coriolis)
      .def_readwrite("safety", &JointImpedanceParams::safety)
      .def_readwrite("friction", &JointImpedanceParams::friction);

  py::class_<CartesianImpedanceBase::Params>(m, "CartesianImpedanceParams")
      .def(py::init<>())
      .def_readwrite("translational_stiffness", &CartesianImpedanceBase::Params::translational_stiffness)
      .def_readwrite("rotational_stiffness", &CartesianImpedanceBase::Params::rotational_stiffness)
      .def_readwrite("translational_error_clip", &CartesianImpedanceBase::Params::translational_error_clip)
      .def_readwrite("rotational_error_clip", &CartesianImpedanceBase::Params::rotational_error_clip)
      .def_readwrite("force_constraints", &CartesianImpedanceBase::Params::force_constraints)
      .def_readwrite("nullspace_target", &CartesianImpedanceBase::Params::nullspace_target)
      .def_readwrite("nullspace_stiffness", &CartesianImpedanceBase::Params::nullspace_stiffness)
      .def_readwrite("safety", &CartesianImpedanceBase::Params::safety)
      .def_readwrite("friction", &CartesianImpedanceBase::Params::friction);

  py::class_<CartesianImpedanceMotion::Params, CartesianImpedanceBase::Params>(m, "CartesianImpedanceMotionParams")
      .def(py::init<>())
      .def_readwrite("target_type", &CartesianImpedanceMotion::Params::target_type)
      .def_readwrite("return_when_finished", &CartesianImpedanceMotion::Params::return_when_finished)
      .def_readwrite("finish_wait_factor", &CartesianImpedanceMotion::Params::finish_wait_factor);

  py::class_<JointImpedanceMotion, Motion<franka::Torques>, std::shared_ptr<JointImpedanceMotion>>(
      m, "JointImpedanceMotion")
      .def(
          py::init<>([](const Vector7d &target,
                        std::optional<Vector7d>
                            target_velocity,
                        std::optional<Vector7d>
                            stiffness,
                        std::optional<Vector7d>
                            damping,
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
          "joint_limit_max_torque"_a = 5.0,
          "friction_coulomb"_a = std::nullopt,
          "friction_viscous"_a = std::nullopt,
          "friction_max_torque"_a = std::nullopt,
          "friction_velocity_epsilon"_a = 0.03)
      .def_property_readonly("target", &JointImpedanceMotion::target)
      .def_property_readonly("target_velocity", &JointImpedanceMotion::target_velocity)
      .def_property_readonly("params", [](const JointImpedanceMotion &m) { return m.params(); });

  py::class_<JointImpedanceTrackingMotion, Motion<franka::Torques>, std::shared_ptr<JointImpedanceTrackingMotion>>(
      m, "JointImpedanceTrackingMotion")
      .def(
          py::init<>([](const std::shared_ptr<JointReferenceHandle> &reference_handle,
                        std::optional<Vector7d>
                            stiffness,
                        std::optional<Vector7d>
                            damping,
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
                        std::shared_ptr<JointImpedanceGainsHandle>
                            gains_handle,
                        double gains_time_constant) {
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
                joint_limit_max_torque,
                friction_coulomb,
                friction_viscous,
                friction_max_torque,
                friction_velocity_epsilon);
            if (gains_handle) {
              return std::make_shared<JointImpedanceTrackingMotion>(
                  reference_handle, params, gains_handle, gains_time_constant);
            }
            return std::make_shared<JointImpedanceTrackingMotion>(reference_handle, params);
          }),
          R"doc(Construct a dynamic joint impedance controller driven by a JointReferenceHandle.

Any constant_torque_offset configured here is added to the per-cycle torque_feedforward values published through the handle.

If gains_handle is provided, the controller reads target gains from it each cycle and exponentially
interpolates toward them with the given time constant, allowing smooth runtime stiffness/damping changes.)doc",
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
          "joint_limit_max_torque"_a = 5.0,
          "friction_coulomb"_a = std::nullopt,
          "friction_viscous"_a = std::nullopt,
          "friction_max_torque"_a = std::nullopt,
          "friction_velocity_epsilon"_a = 0.03,
          "gains_handle"_a = nullptr,
          "gains_time_constant"_a = 0.1)
      .def_property_readonly("target", &JointImpedanceTrackingMotion::target)
      .def_property_readonly("target_velocity", &JointImpedanceTrackingMotion::target_velocity)
      .def_property_readonly("params", [](const JointImpedanceTrackingMotion &m) { return m.params(); });

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
                        std::optional<Vector7d>
                            nullspace_target,
                        double nullspace_stiffness,
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
                        bool return_when_finished,
                        double finish_wait_factor,
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
            params.return_when_finished = return_when_finished;
            params.finish_wait_factor = finish_wait_factor;
            return std::make_shared<CartesianImpedanceMotion>(target, duration, params);
          }),
          R"doc(Construct a Cartesian impedance motion that interpolates to a fixed target pose over the given duration.

Cartesian damping is chosen internally as critically damped with respect to the requested stiffness.

The optional nullspace_target and nullspace_stiffness parameters add a secondary joint-posture objective that is projected into the Jacobian nullspace, so it biases the redundant arm posture without changing the Cartesian task to first order.)doc",
          "target"_a,
          "duration"_a,
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
          "return_when_finished"_a = true,
          "finish_wait_factor"_a = 1.2,
          "friction"_a = std::nullopt)
      .def_property_readonly("target", &CartesianImpedanceMotion::target)
      .def_property_readonly("duration", &CartesianImpedanceMotion::duration)
      .def_property_readonly("params", [](const CartesianImpedanceMotion &m) { return m.params(); });

  py::class_<
      CartesianImpedanceTrackingMotion,
      CartesianImpedanceBase,
      std::shared_ptr<CartesianImpedanceTrackingMotion>>(m, "CartesianImpedanceTrackingMotion")
      .def(
          py::init<>([](const std::shared_ptr<CartesianReferenceHandle> &reference_handle,
                        double translational_stiffness,
                        double rotational_stiffness,
                        std::optional<std::array<std::optional<double>, 6>>
                            force_constraints,
                        std::optional<Vector7d>
                            nullspace_target,
                        double nullspace_stiffness,
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
                        std::shared_ptr<CartesianImpedanceGainsHandle>
                            gains_handle,
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
            if (gains_handle) {
              return std::make_shared<CartesianImpedanceTrackingMotion>(
                  reference_handle, base_params, gains_handle, gains_time_constant);
            }
            return std::make_shared<CartesianImpedanceTrackingMotion>(reference_handle, base_params);
          }),
          R"doc(Construct a dynamic Cartesian impedance tracking controller driven by a CartesianReferenceHandle.

Each published Cartesian reference may optionally include a desired end-effector twist in the base frame.
Cartesian damping is chosen internally as critically damped with respect to the requested stiffness.
The optional nullspace_target and nullspace_stiffness parameters add a secondary joint-posture objective that is projected into the Jacobian nullspace, so it biases the redundant arm posture without changing the Cartesian task to first order.

If gains_handle is provided, the controller reads target gains from it each cycle and exponentially
interpolates toward them with the given time constant, allowing smooth runtime stiffness changes.)doc",
          "reference_handle"_a,
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
          "gains_handle"_a = nullptr,
          "gains_time_constant"_a = 0.1,
          "friction"_a = std::nullopt)
      .def_property_readonly("params", [](const CartesianImpedanceTrackingMotion &m) { return m.params(); });
}
