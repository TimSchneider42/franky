#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "franky.hpp"
#include "util.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the '_a' literal
using namespace franky;

void bind_state_repr(py::module &m) {
  py::class_<Affine>(m, "Affine")
      .def(
          py::init<const Eigen::Matrix<double, 4, 4> &>(),
          "transformation_matrix"_a = Eigen::Matrix<double, 4, 4>::Identity())
      .def(
          py::init<>([](const Eigen::Vector3d &translation, const Eigen::Vector4d &quaternion) {
            return Affine().fromPositionOrientationScale(
                translation, Eigen::Quaterniond(quaternion), Eigen::Vector3d::Ones());
          }),
          "translation"_a = Eigen::Vector3d{0, 0, 0},
          "quaternion"_a = Eigen::Vector4d{0, 0, 0, 1})
      .def(py::init<const Affine &>())  // Copy constructor
      .def(py::self * py::self)
      .def_property_readonly("inverse", [](const Affine &affine) { return affine.inverse(); })
      .def_property_readonly("translation", [](const Affine &affine) { return affine.translation(); })
      .def_property_readonly(
          "quaternion", [](const Affine &affine) { return Eigen::Quaterniond(affine.rotation()).coeffs(); })
      .def_property_readonly("matrix", [](const Affine &affine) { return affine.matrix(); })
      .def("__repr__", strFromStream<Affine>)
      .def(
          py::pickle(
              [](const Affine &affine) {  // __getstate__
                return py::make_tuple(affine.translation(), Eigen::Quaterniond(affine.rotation()).coeffs());
              },
              [](const py::tuple &t) {  // __setstate__
                if (t.size() != 2) throw std::runtime_error("Invalid state!");
                return Affine().fromPositionOrientationScale(
                    t[0].cast<Eigen::Vector3d>(),
                    Eigen::Quaterniond(t[1].cast<Eigen::Vector4d>()),
                    Eigen::Vector3d::Ones());
              }));

  py::class_<ElbowState>(m, "ElbowState")
      .def(
          py::init<double, std::optional<FlipDirection>>(),
          "joint_3_pos"_a,
          "joint_4_flip"_a = std::nullopt)
      .def(py::init<const ElbowState &>())  // Copy constructor
      .def_property_readonly("joint_3_pos", &ElbowState::joint_3_pos)
      .def_property_readonly("joint_4_flip", &ElbowState::joint_4_flip)
      .def("__repr__", &strFromStream<ElbowState>)
      .def(
          py::pickle(
              [](const ElbowState &elbow_state) {  // __getstate__
                return py::make_tuple(elbow_state.joint_3_pos(), elbow_state.joint_4_flip());
              },
              [](const py::tuple &t) {  // __setstate__
                if (t.size() != 2) throw std::runtime_error("Invalid state!");
                return ElbowState(t[0].cast<double>(), t[1].cast<std::optional<FlipDirection>>());
              }));

  py::class_<Twist>(m, "Twist")
      .def(
          py::init([](const std::optional<Eigen::Vector3d> &linear_velocity,
                      const std::optional<Eigen::Vector3d> &angular_velocity) {
            return Twist(
                linear_velocity.value_or(Eigen::Vector3d::Zero()), angular_velocity.value_or(Eigen::Vector3d::Zero()));
          }),
          "linear_velocity"_a = std::nullopt,
          "angular_velocity"_a = std::nullopt)
      .def(py::init<const Twist &>())  // Copy constructor
      .def("propagate_through_link", &Twist::propagateThroughLink, "link_translation"_a)
      .def(
          "transform_with",
          [](const Twist &twist, const Affine &affine) { return twist.transformWith(affine); },
          "affine"_a)
      .def(
          "transform_with",
          [](const Twist &twist, const Eigen::Vector4d &quaternion) {
            return twist.transformWith(Eigen::Quaterniond(quaternion));
          },
          "quaternion"_a)
      .def_property_readonly("linear", &Twist::linear_velocity)
      .def_property_readonly("angular", &Twist::angular_velocity)
      .def(
          "__rmul__", [](const Twist &twist, const Affine &affine) { return affine * twist; }, py::is_operator())
      .def(
          "__rmul__",
          [](const Twist &twist, const Eigen::Vector4d &quaternion) { return Eigen::Quaterniond(quaternion) * twist; },
          py::is_operator())
      .def("__repr__", &strFromStream<Twist>)
      .def(
          py::pickle(
              [](const Twist &twist) {  // __getstate__
                return py::make_tuple(twist.linear_velocity(), twist.angular_velocity());
              },
              [](const py::tuple &t) {  // __setstate__
                if (t.size() != 2) throw std::runtime_error("Invalid state!");
                return Twist(t[0].cast<Eigen::Vector3d>(), t[1].cast<Eigen::Vector3d>());
              }));

  py::class_<TwistAcceleration>(m, "TwistAcceleration")
      .def(
          py::init([](const std::optional<Eigen::Vector3d> &linear_acceleration,
                      const std::optional<Eigen::Vector3d> &angular_acceleration) {
            return TwistAcceleration(
                linear_acceleration.value_or(Eigen::Vector3d::Zero()),
                angular_acceleration.value_or(Eigen::Vector3d::Zero()));
          }),
          "linear_acceleration"_a = std::nullopt,
          "angular_acceleration"_a = std::nullopt)
      .def(py::init<const TwistAcceleration &>())  // Copy constructor
      .def(
          "propagate_through_link",
          &TwistAcceleration::propagateThroughLink,
          "link_translation"_a,
          "base_angular_acceleration"_a)
      .def(
          "transform_with",
          [](const TwistAcceleration &twist, const Affine &affine) { return twist.transformWith(affine); },
          "affine"_a)
      .def(
          "transform_with",
          [](const TwistAcceleration &twist, const Eigen::Vector4d &quaternion) {
            return twist.transformWith(Eigen::Quaterniond(quaternion));
          },
          "quaternion"_a)
      .def_property_readonly("linear", &TwistAcceleration::linear_acceleration)
      .def_property_readonly("angular", &TwistAcceleration::angular_acceleration)
      .def(
          "__rmul__",
          [](const TwistAcceleration &twist, const Affine &affine) { return affine * twist; },
          py::is_operator())
      .def(
          "__rmul__",
          [](const TwistAcceleration &twist, const Eigen::Vector4d &quaternion) {
            return Eigen::Quaterniond(quaternion) * twist;
          },
          py::is_operator())
      .def("__repr__", &strFromStream<TwistAcceleration>)
      .def(
          py::pickle(
              [](const TwistAcceleration &twist) {  // __getstate__
                return py::make_tuple(twist.linear_acceleration(), twist.angular_acceleration());
              },
              [](const py::tuple &t) {  // __setstate__
                if (t.size() != 2) throw std::runtime_error("Invalid state!");
                return TwistAcceleration(t[0].cast<Eigen::Vector3d>(), t[1].cast<Eigen::Vector3d>());
              }));

  py::class_<RobotPose>(m, "RobotPose")
      .def(
          py::init<Affine, std::optional<ElbowState>>(),
          "end_effector_pose"_a,
          "elbow_state"_a = std::nullopt)
      .def(py::init<const RobotPose &>())  // Copy constructor
      .def("change_end_effector_frame", &RobotPose::changeEndEffectorFrame, "offset_world_frame"_a)
      .def("with_elbow_state", &RobotPose::withElbowState, "elbow_state"_a)
      .def_property_readonly("end_effector_pose", &RobotPose::end_effector_pose)
      .def_property_readonly("elbow_state", &RobotPose::elbow_state)
      .def(
          "__mul__",
          [](const RobotPose &robot_pose, const Affine &affine) { return robot_pose * affine; },
          py::is_operator())
      .def(
          "__rmul__",
          [](const RobotPose &robot_pose, const Affine &affine) { return affine * robot_pose; },
          py::is_operator())
      .def("__repr__", strFromStream<RobotPose>)
      .def(
          py::pickle(
              [](const RobotPose &robot_pose) {  // __getstate__
                return py::make_tuple(robot_pose.end_effector_pose(), robot_pose.elbow_state());
              },
              [](const py::tuple &t) {  // __setstate__
                if (t.size() != 2) throw std::runtime_error("Invalid state!");
                return RobotPose(t[0].cast<Affine>(), t[1].cast<std::optional<ElbowState>>());
              }));
  py::implicitly_convertible<Affine, RobotPose>();

  py::class_<RobotVelocity>(m, "RobotVelocity")
      .def(
          py::init<Twist, std::optional<double>>(),
          "end_effector_twist"_a,
          "elbow_velocity"_a = std::nullopt)
      .def(py::init<const RobotVelocity &>())  // Copy constructor
      .def("change_end_effector_frame", &RobotVelocity::changeEndEffectorFrame, "offset_world_frame"_a)
      .def("with_elbow_velocity", &RobotVelocity::withElbowVelocity, "elbow_velocity"_a)
      .def_property_readonly("end_effector_twist", &RobotVelocity::end_effector_twist)
      .def_property_readonly("elbow_velocity", &RobotVelocity::elbow_velocity)
      .def(
          "__rmul__",
          [](const RobotVelocity &robot_velocity, const Affine &affine) { return affine * robot_velocity; },
          py::is_operator())
      .def(
          "__rmul__",
          [](const RobotVelocity &robot_velocity, const Eigen::Vector4d &quaternion) {
            return Eigen::Quaterniond(quaternion) * robot_velocity;
          },
          py::is_operator())
      .def("__repr__", strFromStream<RobotVelocity>)
      .def(
          py::pickle(
              [](const RobotVelocity &robot_velocity) {  // __getstate__
                return py::make_tuple(robot_velocity.end_effector_twist(), robot_velocity.elbow_velocity());
              },
              [](const py::tuple &t) {  // __setstate__
                if (t.size() != 2) throw std::runtime_error("Invalid state!");
                return RobotVelocity(t[0].cast<Twist>(), t[1].cast<double>());
              }));
  py::implicitly_convertible<Twist, RobotVelocity>();

  py::class_<CartesianState>(m, "CartesianState")
      .def(py::init<const RobotPose &>(), "pose"_a)
      .def(
          py::init<const RobotPose &, const RobotVelocity &>(),
          "pose"_a,
          "velocity"_a)
      .def(py::init<const CartesianState &>())  // Copy constructor
      .def("transform_with", &CartesianState::transformWith, "transform"_a)
      .def("change_end_effector_frame", &CartesianState::changeEndEffectorFrame, "transform"_a)
      .def_property_readonly("pose", &CartesianState::pose)
      .def_property_readonly("velocity", &CartesianState::velocity)
      .def(
          "__rmul__",
          [](const CartesianState &cartesian_state, const Affine &affine) { return affine * cartesian_state; },
          py::is_operator())
      .def("__repr__", strFromStream<CartesianState>)
      .def(
          py::pickle(
              [](const CartesianState &cartesian_state) {  // __getstate__
                return py::make_tuple(cartesian_state.pose(), cartesian_state.velocity());
              },
              [](const py::tuple &t) {  // __setstate__
                if (t.size() != 2) throw std::runtime_error("Invalid state!");
                return CartesianState(t[0].cast<RobotPose>(), t[1].cast<RobotVelocity>());
              }));
  py::implicitly_convertible<RobotPose, CartesianState>();
  py::implicitly_convertible<Affine, CartesianState>();

  py::class_<JointState>(m, "JointState")
      .def(py::init<const Vector7d &>(), "position"_a)
      .def(
          py::init<const Vector7d &, const Vector7d &>(),
          "position"_a,
          "velocity"_a)
      .def(py::init<const JointState &>())  // Copy constructor
      .def_property_readonly("position", &JointState::position)
      .def_property_readonly("velocity", &JointState::velocity)
      .def("__repr__", strFromStream<JointState>)
      .def(
          py::pickle(
              [](const JointState &joint_state) {  // __getstate__
                return py::make_tuple(joint_state.position(), joint_state.velocity());
              },
              [](const py::tuple &t) {  // __setstate__
                if (t.size() != 2) throw std::runtime_error("Invalid state!");
                return JointState(t[0].cast<Vector7d>(), t[1].cast<Vector7d>());
              }));
  py::implicitly_convertible<Vector7d, JointState>();
  py::implicitly_convertible<std::array<double, 7>, JointState>();

  py::class_<franka::Torques>(m, "Torques")
      .def_readonly("tau_J", &franka::Torques::tau_J)
      .def(
          py::pickle(
              [](const franka::Torques &torques) {  // __getstate__
                return py::make_tuple(torques.tau_J);
              },
              [](const py::tuple &t) {  // __setstate__
                if (t.size() != 1) throw std::runtime_error("Invalid state!");
                return franka::Torques(t[0].cast<std::array<double, 7>>());
              }));

  py::class_<franka::JointVelocities>(m, "JointVelocities")
      .def_readonly("dq", &franka::JointVelocities::dq)
      .def(
          py::pickle(
              [](const franka::JointVelocities &velocities) {  // __getstate__
                return py::make_tuple(velocities.dq);
              },
              [](const py::tuple &t) {  // __setstate__
                if (t.size() != 1) throw std::runtime_error("Invalid state!");
                return franka::JointVelocities(t[0].cast<std::array<double, 7>>());
              }));

  py::class_<franka::JointPositions>(m, "JointPositions")
      .def_readonly("q", &franka::JointPositions::q)
      .def(
          py::pickle(
              [](const franka::JointPositions &positions) {  // __getstate__
                return py::make_tuple(positions.q);
              },
              [](const py::tuple &t) {  // __setstate__
                if (t.size() != 1) throw std::runtime_error("Invalid state!");
                return franka::JointPositions(t[0].cast<std::array<double, 7>>());
              }));

  py::class_<franka::CartesianVelocities>(m, "CartesianVelocities")
      .def_readonly("O_dP_EE", &franka::CartesianVelocities::O_dP_EE)
      .def_readonly("elbow", &franka::CartesianVelocities::elbow)
      .def(
          py::pickle(
              [](const franka::CartesianVelocities &velocities) {  // __getstate__
                return py::make_tuple(velocities.O_dP_EE, velocities.elbow);
              },
              [](const py::tuple &t) {  // __setstate__
                if (t.size() != 2) throw std::runtime_error("Invalid state!");
                return franka::CartesianVelocities(
                    t[0].cast<std::array<double, 6>>(), t[1].cast<std::array<double, 2>>());
              }));

  py::class_<franka::CartesianPose>(m, "CartesianPose")
      .def_readonly("O_T_EE", &franka::CartesianPose::O_T_EE)
      .def(
          py::pickle(
              [](const franka::CartesianPose &pose) {  // __getstate__
                return py::make_tuple(pose.O_T_EE, pose.elbow);
              },
              [](const py::tuple &t) {  // __setstate__
                if (t.size() != 2) throw std::runtime_error("Invalid state!");
                return franka::CartesianPose(t[0].cast<std::array<double, 16>>(), t[1].cast<std::array<double, 2>>());
              }));
}
