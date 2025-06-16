#include <franka/gripper_state.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "franky.hpp"
#include "macros.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the '_a' literal
using namespace franky;

#define ADD_FIELD_RO(obj_type, unused, name) .def_readonly(#name, &obj_type::name)
#define ADD_FIELDS_RO(obj, obj_type, ...) obj MAP_C1(ADD_FIELD_RO, obj_type, __VA_ARGS__);

#ifdef FRANKA_0_8
#define EXTRA_FIELDS1 F_T_NE, NE_T_EE,
#else
#define EXTRA_FIELDS1
#endif

#ifdef FRANKA_0_9
#define EXTRA_FIELDS2 O_ddP_O,
#else
#define EXTRA_FIELDS2
#endif

#define ROBOT_STATE_FIELDS                                                                                          \
  O_T_EE, O_T_EE_d, F_T_EE, EXTRA_FIELDS1 EE_T_K, m_ee, I_ee, F_x_Cee, m_load, I_load, F_x_Cload, m_total, I_total, \
      F_x_Ctotal, elbow, elbow_d, elbow_c, delbow_c, ddelbow_c, tau_J, tau_J_d, dtau_J, q, q_d, dq, dq_d, ddq_d,    \
      joint_contact, cartesian_contact, joint_collision, cartesian_collision, tau_ext_hat_filtered, O_F_ext_hat_K,  \
      K_F_ext_hat_K, O_dP_EE_d, EXTRA_FIELDS2 O_T_EE_c, O_dP_EE_c, O_ddP_EE_c, theta, dtheta, current_errors,       \
      last_motion_errors, control_command_success_rate, robot_mode, time, q_est, dq_est, ddq_est, O_dP_EE_est,      \
      O_ddP_EE_est, delbow_est, ddelbow_est

#define GRIPPER_STATE_FIELDS width, max_width, is_grasped, temperature, time

void bind_robot_state(py::module &m) {
  py::class_<RobotState> robot_state(m, "RobotState");
  ADD_FIELDS_RO(robot_state, RobotState, ROBOT_STATE_FIELDS)
  robot_state.def(
      py::pickle(
          [](const RobotState &state) {  // __getstate__
            return PACK_TUPLE(state, ROBOT_STATE_FIELDS);
          },
          [](const py::tuple &t) {  // __setstate__
            if (t.size() != COUNT(ROBOT_STATE_FIELDS)) throw std::runtime_error("Invalid state!");
            return UNPACK_TUPLE(RobotState, t, ROBOT_STATE_FIELDS);
          }));

  py::class_<franka::GripperState> gripper_state(m, "GripperState");
  ADD_FIELDS_RO(gripper_state, franka::GripperState, GRIPPER_STATE_FIELDS)
  gripper_state.def(
      py::pickle(
          [](const franka::GripperState &state) {  // __getstate__
            return PACK_TUPLE(state, GRIPPER_STATE_FIELDS);
          },
          [](const py::tuple &t) {  // __setstate__
            if (t.size() != COUNT(GRIPPER_STATE_FIELDS)) throw std::runtime_error("Invalid state!");
            return UNPACK_TUPLE(franka::GripperState, t, GRIPPER_STATE_FIELDS);
          }));
}
