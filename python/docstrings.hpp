/*
  This file contains docstrings for use in the Python bindings.
  Do not edit! They were automatically extracted by pybind11_mkdoc.

  The docstrings of the franka::* entities are extracted from the headers of libfranka
  (https://github.com/frankaemika/libfranka), Copyright (c) Franka Robotics GmbH, which is
  licensed under the Apache License, Version 2.0
  (http://www.apache.org/licenses/LICENSE-2.0).
 */

#define MKD_EXPAND(x) x
#define MKD_COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...) COUNT
#define MKD_VA_SIZE(...) MKD_EXPAND(MKD_COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1, 0))
#define MKD_CAT1(a, b) a##b
#define MKD_CAT2(a, b) MKD_CAT1(a, b)
#define MKD_DOC1(n1) mkd_doc_##n1
#define MKD_DOC2(n1, n2) mkd_doc_##n1##_##n2
#define MKD_DOC3(n1, n2, n3) mkd_doc_##n1##_##n2##_##n3
#define MKD_DOC4(n1, n2, n3, n4) mkd_doc_##n1##_##n2##_##n3##_##n4
#define MKD_DOC5(n1, n2, n3, n4, n5) mkd_doc_##n1##_##n2##_##n3##_##n4##_##n5
#define MKD_DOC6(n1, n2, n3, n4, n5, n6) mkd_doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define MKD_DOC7(n1, n2, n3, n4, n5, n6, n7) mkd_doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define DOC(...) MKD_EXPAND(MKD_EXPAND(MKD_CAT2(MKD_DOC, MKD_VA_SIZE(__VA_ARGS__)))(__VA_ARGS__))

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

static const char *mkd_doc_franka_ActiveControl = R"doc(Documented in ActiveControlBase)doc";

static const char *mkd_doc_franka_ActiveControlBase =
    R"doc(Allows the user to read the state of a Robot and to send new control commands after starting a
control process of a Robot.

hint: To create an ActiveControlBase, see franka::ActiveTorqueControl or
franka::ActiveMotionGenerator)doc";

static const char *mkd_doc_franka_ActiveControlBase_2 = R"doc()doc";

static const char *mkd_doc_franka_ActiveControlBase_ActiveControlBase = R"doc()doc";

static const char *mkd_doc_franka_ActiveControlBase_readOnce =
    R"doc(Waits for a robot state update and returns it.

Returns:
    Current robot state & time since last read operation

Raises:
    NetworkException: if the connection is lost, e.g. after a timeout.
    ProtocolException: if robot returns an unexpected message.
    ControlException: if robot is in an error state.

)doc";

static const char *mkd_doc_franka_ActiveControlBase_writeOnce =
    R"doc(Updates torque commands of an active control

hint: implemented in ActiveTorqueControl

)doc";

static const char *mkd_doc_franka_ActiveControlBase_writeOnce_2 =
    R"doc(Updates the joint position and torque commands of an active control

hint: implemented in ActiveMotionGenerator<JointPositions>

)doc";

static const char *mkd_doc_franka_ActiveControlBase_writeOnce_3 =
    R"doc(Updates the joint velocity and torque commands of an active control

hint: implemented in ActiveMotionGenerator<JointVelocities>

)doc";

static const char *mkd_doc_franka_ActiveControlBase_writeOnce_4 =
    R"doc(Updates the cartesian position and torque commands of an active control

hint: implemented in ActiveMotionGenerator<CartesianPose>

)doc";

static const char *mkd_doc_franka_ActiveControlBase_writeOnce_5 =
    R"doc(Updates the cartesian velocity and torque commands of an active control

hint: implemented in ActiveMotionGenerator<CartesianVelocities>

)doc";

static const char *mkd_doc_franka_ActiveControlBase_writeOnce_6 =
    R"doc(Updates the joint position commands of an active control, with internal controller

Args:
    motion_generator_input: the new motion generator input

)doc";

static const char *mkd_doc_franka_ActiveControlBase_writeOnce_7 =
    R"doc(Updates the joint velocity commands of an active control, with internal controller

Args:
    motion_generator_input: the new motion generator input

)doc";

static const char *mkd_doc_franka_ActiveControlBase_writeOnce_8 =
    R"doc(Updates the cartesian pose commands of an active control, with internal controller

Args:
    motion_generator_input: the new motion generator input

)doc";

static const char *mkd_doc_franka_ActiveControlBase_writeOnce_9 =
    R"doc(Updates the cartesian velocity commands of an active control, with internal controller

Args:
    motion_generator_input: the new motion generator input

)doc";

static const char *mkd_doc_franka_ActiveControl_ActiveControl =
    R"doc(Construct a new ActiveControl object

Args:
    robot_impl: shared_ptr to the Robot::Impl in the Robot
    motion_id: id of the managed motion
    control_lock: of the Robot, preventing other read and write accesses during the active control

)doc";

static const char *mkd_doc_franka_ActiveControl_control_finished =
    R"doc(flag indicating if control process is finished)doc";

static const char *mkd_doc_franka_ActiveControl_control_lock =
    R"doc(control-lock preventing parallel control processes)doc";

static const char *mkd_doc_franka_ActiveControl_last_read_access = R"doc(duration to last read access)doc";

static const char *mkd_doc_franka_ActiveControl_motion_id = R"doc(motion id of running motion)doc";

static const char *mkd_doc_franka_ActiveControl_readOnce = R"doc()doc";

static const char *mkd_doc_franka_ActiveControl_robot_impl =
    R"doc(shared pointer to Robot::Impl instance for read and write accesses)doc";

static const char *mkd_doc_franka_ActiveControl_writeOnce = R"doc()doc";

static const char *mkd_doc_franka_ActiveControl_writeOnce_2 = R"doc()doc";

static const char *mkd_doc_franka_ActiveControl_writeOnce_3 = R"doc()doc";

static const char *mkd_doc_franka_ActiveControl_writeOnce_4 = R"doc()doc";

static const char *mkd_doc_franka_ActiveControl_writeOnce_5 = R"doc()doc";

static const char *mkd_doc_franka_ActiveControl_writeOnce_6 = R"doc()doc";

static const char *mkd_doc_franka_ActiveControl_writeOnce_7 = R"doc()doc";

static const char *mkd_doc_franka_ActiveControl_writeOnce_8 = R"doc()doc";

static const char *mkd_doc_franka_ActiveControl_writeOnce_9 = R"doc()doc";

static const char *mkd_doc_franka_ActiveControl_wrong_write_once_method_called = R"doc()doc";

static const char *mkd_doc_franka_ActiveMotionGenerator =
    R"doc(Allows the user to read the state of a Robot and to send new motion generator commands after
starting a control process of a Robot.

hint: To create an ActiveMotionGenerator, see franka::Robot)doc";

static const char *mkd_doc_franka_ActiveMotionGenerator_ActiveMotionGenerator =
    R"doc(Construct a new ActiveMotionGenerator object

Args:
    robot: shared_ptr to the Robot::Impl in the Robot
    motion_id: id of the managed motion
    control_lock: of the Robot, preventing other read and write accesses during the active control
    controller_type: defining which controller shall be used

)doc";

static const char *mkd_doc_franka_ActiveMotionGenerator_controller_type = R"doc()doc";

static const char *mkd_doc_franka_ActiveMotionGenerator_isTorqueControlFinished = R"doc()doc";

static const char *mkd_doc_franka_ActiveMotionGenerator_writeOnce =
    R"doc( Updates the motion generator commands of an active control

Args:
    motion_generator_input: the new motion generator input
    control_input: optional: the external control input for each joint, if an external controller is
                   used

Raises:
    ControlException: if an error related to torque control or motion generation occurred, or the
                      motion was already finished.
    NetworkException: if the connection is lost, e.g. after a timeout.

)doc";

static const char *mkd_doc_franka_ActiveTorqueControl =
    R"doc(Allows the user to read the state of a Robot and to send new torque control commands after starting
a control process of a Robot.

hint: To create an ActiveTorqueControl, see franka::Robot)doc";

static const char *mkd_doc_franka_ActiveTorqueControl_ActiveTorqueControl =
    R"doc(Construct a new ActiveTorqueControl object

Args:
    robot: shared_ptr to the Robot::Impl in the Robot
    motion_id: id of the managed motion
    control_lock: of the Robot, preventing other read and write accesses during the active control

)doc";

static const char *mkd_doc_franka_ActiveTorqueControl_writeOnce =
    R"doc(Updates the joint-level based torque commands of an active joint effort control

Args:
    control_input: the new joint-level based torques

Raises:
    ControlException: if an error related to torque control or motion generation occurred, or the
                      motion was already finished.
    NetworkException: if the connection is lost, e.g. after a timeout.

)doc";

static const char *mkd_doc_franka_CartesianPose = R"doc(Stores values for Cartesian pose motion generation.)doc";

static const char *mkd_doc_franka_CartesianPose_CartesianPose =
    R"doc(Creates a new CartesianPose instance.

Args:
    cartesian_pose: Desired vectorized homogeneous transformation matrix :math:`^O
                         {\mathbf{T}_{EE}}_{d}`, column major, that transforms from the end effector
                         frame :math:`EE` to base frame :math:`O`. Equivalently, it is the desired
                         end effector pose in base frame.

)doc";

static const char *mkd_doc_franka_CartesianPose_CartesianPose_2 =
    R"doc(Creates a new CartesianPose instance.

Args:
    cartesian_pose: Desired vectorized homogeneous transformation matrix :math:`^O
                         {\mathbf{T}_{EE}}_{d}`, column major, that transforms from the end effector
                         frame :math:`EE` to base frame :math:`O`. Equivalently, it is the desired
                         end effector pose in base frame.
    elbow: Elbow configuration (see elbow member for more details).

)doc";

static const char *mkd_doc_franka_CartesianPose_CartesianPose_3 =
    R"doc(Creates a new CartesianPose instance.

Args:
    cartesian_pose: Desired vectorized homogeneous transformation matrix :math:`^O
                         {\mathbf{T}_{EE}}_{d}`, column major, that transforms from the end effector
                         frame :math:`EE` to base frame :math:`O`. Equivalently, it is the desired
                         end effector pose in base frame.

Raises:
    std::invalid_argument: if the given initializer list has an invalid number of arguments.

)doc";

static const char *mkd_doc_franka_CartesianPose_CartesianPose_4 =
    R"doc(Creates a new CartesianPose instance.

Args:
    cartesian_pose: Desired vectorized homogeneous transformation matrix :math:`^O
                         {\mathbf{T}_{EE}}_{d}`, column major, that transforms from the end effector
                         frame :math:`EE` to base frame :math:`O`. Equivalently, it is the desired
                         end effector pose in base frame.
    elbow: Elbow configuration (see elbow member for more details).

Raises:
    std::invalid_argument: if a given initializer list has an invalid number of arguments.

)doc";

static const char *mkd_doc_franka_CartesianPose_O_T_EE =
    R"doc(Homogeneous transformation :math:`^O{\mathbf{T}_{EE}}_{d}`, column major, that transforms from the
end effector frame :math:`EE` to base frame :math:`O`. Equivalently, it is the desired end effector
pose in base frame.)doc";

static const char *mkd_doc_franka_CartesianPose_elbow =
    R"doc(Elbow configuration.

The values of the array are:
 - elbow[0]: Position of the 3rd joint in :math:`[rad]`. - elbow[1]: Flip direction of the elbow
   (4th joint):
   - +1 if :math:`q_4 > q_{elbow-flip}` - 0 if :math:`q_4 == q_{elbow-flip} ` - -1 if :math:`q_4 <
     q_{elbow-flip} `
   . with :math:`q_{elbow-flip}` as specified in the robot interface specification page in the FCI
Documentation.)doc";

static const char *mkd_doc_franka_CartesianPose_hasElbow =
    R"doc(Determines whether there is a stored elbow configuration.

Returns:
    True if there is a stored elbow configuration, false otherwise.

)doc";

static const char *mkd_doc_franka_CartesianVelocities =
    R"doc(Stores values for Cartesian velocity motion generation.

The Cartesian velocity of the end-effector is expressed in a frame parallel to the fixed/base frame,
whose origin is the same as the end-effector frame. Rotations are thus expressed around the end-
effector and parallel to the base frame.)doc";

static const char *mkd_doc_franka_CartesianVelocities_CartesianVelocities =
    R"doc(Creates a new CartesianVelocities instance.

Args:
    cartesian_velocities: Desired Cartesian velocity with respect to the o-frame "base frame O"
                               with :math:`(\dot x, \dot y, \dot z)` in :math:`[m/s]` and
                               :math:`(\omega_x, \omega_y, \omega_z)` in :math:`[rad/s]`.

)doc";

static const char *mkd_doc_franka_CartesianVelocities_CartesianVelocities_2 =
    R"doc(Creates a new CartesianVelocities instance.

Args:
    cartesian_velocities: Desired Cartesian velocity with respect to the o-frame "base frame O"
                               with :math:`(\dot x, \dot y, \dot z)` in :math:`[m/s]` and
                               :math:`(\omega_x, \omega_y, \omega_z)` in :math:`[rad/s]`.
    elbow: Elbow configuration (see elbow member for more details).

)doc";

static const char *mkd_doc_franka_CartesianVelocities_CartesianVelocities_3 =
    R"doc(Creates a new CartesianVelocities instance.

Args:
    cartesian_velocities: Desired Cartesian velocity with respect to the o-frame "base frame O"
                               with :math:`(\dot x, \dot y, \dot z)` in :math:`[m/s]` and
                               :math:`(\omega_x, \omega_y, \omega_z)` in :math:`[rad/s]`.

Raises:
    std::invalid_argument: if the given initializer list has an invalid number of arguments.

)doc";

static const char *mkd_doc_franka_CartesianVelocities_CartesianVelocities_4 =
    R"doc(Creates a new CartesianVelocities instance.

Args:
    cartesian_velocities: Desired Cartesian velocity with respect to the o-frame "base frame O"
                               with :math:`(\dot x, \dot y, \dot z)` in :math:`[m/s]` and
                               :math:`(\omega_x, \omega_y, \omega_z)` in :math:`[rad/s]`.
    elbow: Elbow configuration (see elbow member for more details).

Raises:
    std::invalid_argument: if a given initializer list has an invalid number of arguments.

)doc";

static const char *mkd_doc_franka_CartesianVelocities_O_dP_EE =
    R"doc(Cartesian velocity with respect to the o-frame "base frame O" with :math:`(\dot x, \dot y, \dot z)`
in :math:`[m/s]` and :math:`(\omega_x, \omega_y, \omega_z)` in :math:`[rad/s]`.)doc";

static const char *mkd_doc_franka_CartesianVelocities_elbow =
    R"doc(Elbow configuration.

The values of the array are:
 - elbow[0]: Position of the 3rd joint in :math:`[rad]`. - elbow[1]: Flip direction of the elbow
   (4th joint):
   - +1 if :math:`q_4 > \alpha` - 0 if :math:`q_4 == \alpha ` - -1 if :math:`q_4 < \alpha `
   . with :math:`\alpha = -0.467002423653011` :math:`rad`)doc";

static const char *mkd_doc_franka_CartesianVelocities_hasElbow =
    R"doc(Determines whether there is a stored elbow configuration.

Returns:
    True if there is a stored elbow configuration, false otherwise.

)doc";

static const char *mkd_doc_franka_CommandException =
    R"doc(CommandException is thrown if an error occurs during command execution.)doc";

static const char *mkd_doc_franka_ControlException =
    R"doc(ControlException is thrown if an error occurs during motion generation or torque control. The
exception holds a vector with the last received robot states. The number of recorded states can be
configured in the Robot constructor.)doc";

static const char *mkd_doc_franka_ControlException_ControlException =
    R"doc(Creates the exception with an explanatory string and a Log object.

Args:
    what: Explanatory string.
    log: Vector of last received states and commands.

)doc";

static const char *mkd_doc_franka_ControlException_log =
    R"doc(Vector of states and commands logged just before the exception occurred.)doc";

static const char *mkd_doc_franka_ControllerMode = R"doc(Available controller modes for a franka::Robot.)doc";

static const char *mkd_doc_franka_ControllerMode_kCartesianImpedance = R"doc()doc";

static const char *mkd_doc_franka_ControllerMode_kJointImpedance = R"doc()doc";

static const char *mkd_doc_franka_Duration = R"doc(Represents a duration with millisecond resolution.)doc";

static const char *mkd_doc_franka_Duration_Duration =
    R"doc(Creates a new Duration instance with zero milliseconds.)doc";

static const char *mkd_doc_franka_Duration_Duration_2 =
    R"doc(Creates a new Duration instance from the given number of milliseconds.

Args:
    milliseconds: Number of milliseconds.

)doc";

static const char *mkd_doc_franka_Duration_Duration_3 =
    R"doc(Creates a new Duration instance from an std::chrono::duration.

Args:
    duration: Duration.

)doc";

static const char *mkd_doc_franka_Duration_Duration_4 = R"doc(Creates a copy of a Duration instance.)doc";

static const char *mkd_doc_franka_Duration_duration = R"doc(@})doc";

static const char *mkd_doc_franka_Duration_operator_add =
    R"doc(Performs addition.

Args:
    rhs: Right-hand side of the operation.

Returns:
    Result of the operation.

)doc";

static const char *mkd_doc_franka_Duration_operator_assign =
    R"doc(Assigns the contents of one Duration to another.

Returns:
    Result of the operation.

)doc";

static const char *mkd_doc_franka_Duration_operator_div =
    R"doc(Performs division.

Args:
    rhs: Right-hand side of the operation.

Returns:
    Result of the operation.

)doc";

static const char *mkd_doc_franka_Duration_operator_div_2 =
    R"doc(Performs division.

Args:
    rhs: Right-hand side of the operation.

Returns:
    Result of the operation.

)doc";

static const char *mkd_doc_franka_Duration_operator_duration =
    R"doc(Returns the stored duration as an std::chrono::duration.

Returns:
    Duration as std::chrono::duration.

)doc";

static const char *mkd_doc_franka_Duration_operator_eq =
    R"doc(Compares two durations for equality.

Args:
    rhs: Right-hand side of the comparison.

Returns:
    True if the duration are equal, false otherwise.

)doc";

static const char *mkd_doc_franka_Duration_operator_ge =
    R"doc(Compares two durations.

Args:
    rhs: Right-hand side of the comparison.

Returns:
    True if this duration is longer than or equal to rhs, false otherwise.

)doc";

static const char *mkd_doc_franka_Duration_operator_gt =
    R"doc(Compares two durations.

Args:
    rhs: Right-hand side of the comparison.

Returns:
    True if this duration is longer than rhs, false otherwise.

)doc";

static const char *mkd_doc_franka_Duration_operator_iadd =
    R"doc(Performs addition.

Args:
    rhs: Right-hand side of the operation.

Returns:
    This duration.

)doc";

static const char *mkd_doc_franka_Duration_operator_idiv =
    R"doc(Performs division.

Args:
    rhs: Right-hand side of the operation.

Returns:
    This duration.

)doc";

static const char *mkd_doc_franka_Duration_operator_imod =
    R"doc(Performs the modulo operation.

Args:
    rhs: Right-hand side of the operation.

Returns:
    This duration.

)doc";

static const char *mkd_doc_franka_Duration_operator_imod_2 =
    R"doc(Performs the modulo operation.

Args:
    rhs: Right-hand side of the operation.

Returns:
    This duration.

)doc";

static const char *mkd_doc_franka_Duration_operator_imul =
    R"doc(Performs multiplication.

Args:
    rhs: Right-hand side of the operation.

Returns:
    This duration.

)doc";

static const char *mkd_doc_franka_Duration_operator_isub =
    R"doc(Performs subtraction.

Args:
    rhs: Right-hand side of the operation.

Returns:
    This duration.

)doc";

static const char *mkd_doc_franka_Duration_operator_le =
    R"doc(Compares two durations.

Args:
    rhs: Right-hand side of the comparison.

Returns:
    True if this duration is shorter than or equal to rhs, false otherwise.

)doc";

static const char *mkd_doc_franka_Duration_operator_lt =
    R"doc(Compares two durations.

Args:
    rhs: Right-hand side of the comparison.

Returns:
    True if this duration is shorter than rhs, false otherwise.

)doc";

static const char *mkd_doc_franka_Duration_operator_mod =
    R"doc(Performs the modulo operation.

Args:
    rhs: Right-hand side of the operation.

Returns:
    Result of the operation.

)doc";

static const char *mkd_doc_franka_Duration_operator_mod_2 =
    R"doc(Performs the modulo operation.

Args:
    rhs: Right-hand side of the operation.

Returns:
    Result of the operation.

)doc";

static const char *mkd_doc_franka_Duration_operator_mul =
    R"doc(Performs multiplication.

Args:
    rhs: Right-hand side of the operation.

Returns:
    Result of the operation.

)doc";

static const char *mkd_doc_franka_Duration_operator_ne =
    R"doc(Compares two durations for inequality.

Args:
    rhs: Right-hand side of the comparison.

Returns:
    True if the duration are not equal, false otherwise.

)doc";

static const char *mkd_doc_franka_Duration_operator_sub =
    R"doc(Performs subtraction.

Args:
    rhs: Right-hand side of the operation.

Returns:
    Result of the operation.

)doc";

static const char *mkd_doc_franka_Duration_toMSec =
    R"doc(Returns the stored duration in :math:`[ms]`.

Returns:
    Duration in :math:`[ms]`.

)doc";

static const char *mkd_doc_franka_Duration_toSec =
    R"doc(Returns the stored duration in :math:`[s]`.

Returns:
    Duration in :math:`[s]`.

)doc";

static const char *mkd_doc_franka_Errors =
    R"doc(Enumerates errors that can occur while controlling a franka::Robot.)doc";

static const char *mkd_doc_franka_Errors_Errors = R"doc(Creates an empty Errors instance.)doc";

static const char *mkd_doc_franka_Errors_Errors_2 =
    R"doc(Copy constructs a new Errors instance.

Args:
    other: Other Errors instance.

)doc";

static const char *mkd_doc_franka_Errors_Errors_3 =
    R"doc(Creates a new Errors instance from the given array.

Args:
    errors: Array of error flags.

)doc";

static const char *mkd_doc_franka_Errors_base_acceleration_initialization_timeout =
    R"doc(True if the gravity vector could not be initialized by measureing the base acceleration.)doc";

static const char *mkd_doc_franka_Errors_base_acceleration_invalid_reading =
    R"doc(True if the base acceleration O_ddP_O cannot be determined.)doc";

static const char *mkd_doc_franka_Errors_cartesian_motion_generator_acceleration_discontinuity =
    R"doc(True if commanded acceleration in Cartesian motion generators is discontinuous (target values are
too far apart).)doc";

static const char *mkd_doc_franka_Errors_cartesian_motion_generator_elbow_limit_violation =
    R"doc(True if an external Cartesian motion generator would move into an elbow limit.)doc";

static const char *mkd_doc_franka_Errors_cartesian_motion_generator_elbow_sign_inconsistent =
    R"doc(True if commanded elbow values in Cartesian motion generators are inconsistent.)doc";

static const char *mkd_doc_franka_Errors_cartesian_motion_generator_joint_acceleration_discontinuity =
    R"doc(True if the joint acceleration in Cartesian motion generators is discontinuous after IK calculation.)doc";

static const char *mkd_doc_franka_Errors_cartesian_motion_generator_joint_position_limits_violation =
    R"doc(True if the joint position limits would be exceeded after IK calculation.)doc";

static const char *mkd_doc_franka_Errors_cartesian_motion_generator_joint_velocity_discontinuity =
    R"doc(True if the joint velocity in Cartesian motion generators is discontinuous after IK calculation.)doc";

static const char *mkd_doc_franka_Errors_cartesian_motion_generator_joint_velocity_limits_violation =
    R"doc(True if the joint velocity limits would be exceeded after IK calculation.)doc";

static const char *mkd_doc_franka_Errors_cartesian_motion_generator_start_elbow_invalid =
    R"doc(True if the first elbow value in Cartesian motion generators is too far from initial one.)doc";

static const char *mkd_doc_franka_Errors_cartesian_motion_generator_velocity_discontinuity =
    R"doc(True if commanded velocity in Cartesian motion generators is discontinuous (target values are too
far apart).)doc";

static const char *mkd_doc_franka_Errors_cartesian_motion_generator_velocity_limits_violation =
    R"doc(True if an external Cartesian motion generator would move with too high velocity.)doc";

static const char *mkd_doc_franka_Errors_cartesian_position_limits_violation =
    R"doc(True if the robot moved past any of the virtual walls.)doc";

static const char *mkd_doc_franka_Errors_cartesian_position_motion_generator_invalid_frame =
    R"doc(True if the Cartesian pose is not a valid transformation matrix.)doc";

static const char *mkd_doc_franka_Errors_cartesian_position_motion_generator_start_pose_invalid =
    R"doc(True if an external Cartesian position motion generator was started with a pose too far from the
current pose.)doc";

static const char *mkd_doc_franka_Errors_cartesian_reflex =
    R"doc(True if a collision was detected, i.e. the robot exceeded a torque threshold in a Cartesian motion.)doc";

static const char *mkd_doc_franka_Errors_cartesian_spline_motion_generator_violation =
    R"doc(True if the generated motion violates a joint limit.)doc";

static const char *mkd_doc_franka_Errors_cartesian_velocity_profile_safety_violation =
    R"doc(True if Cartesian velocity profile for internal motions was exceeded.)doc";

static const char *mkd_doc_franka_Errors_cartesian_velocity_violation =
    R"doc(True if the robot exceeded Cartesian velocity limits.)doc";

static const char *mkd_doc_franka_Errors_communication_constraints_violation =
    R"doc(True if minimum network communication quality could not be held during a motion.)doc";

static const char *mkd_doc_franka_Errors_controller_torque_discontinuity =
    R"doc(True if the torque set by the external controller is discontinuous.)doc";

static const char *mkd_doc_franka_Errors_errors = R"doc()doc";

static const char *mkd_doc_franka_Errors_force_control_safety_violation =
    R"doc(True if the robot exceeded safety threshold during force control.)doc";

static const char *mkd_doc_franka_Errors_force_controller_desired_force_tolerance_violation =
    R"doc(True if desired force exceeds the safety thresholds.)doc";

static const char *mkd_doc_franka_Errors_instability_detected = R"doc(True if an instability is detected.)doc";

static const char *mkd_doc_franka_Errors_joint_motion_generator_acceleration_discontinuity =
    R"doc(True if commanded acceleration in joint motion generators is discontinuous (target values are too
far apart).)doc";

static const char *mkd_doc_franka_Errors_joint_motion_generator_position_limits_violation =
    R"doc(True if an external joint motion generator would move into a joint limit.)doc";

static const char *mkd_doc_franka_Errors_joint_motion_generator_velocity_discontinuity =
    R"doc(True if commanded velocity in joint motion generators is discontinuous (target values are too far
apart).)doc";

static const char *mkd_doc_franka_Errors_joint_motion_generator_velocity_limits_violation =
    R"doc(True if an external joint motion generator exceeded velocity limits.)doc";

static const char *mkd_doc_franka_Errors_joint_move_in_wrong_direction =
    R"doc(True if the robot is in joint position limits violation error and the user guides the robot further
towards the limit.)doc";

static const char *mkd_doc_franka_Errors_joint_p2p_insufficient_torque_for_planning =
    R"doc(True if the robot is overloaded for the required motion.

Applies only to motions started from Desk.)doc";

static const char *mkd_doc_franka_Errors_joint_position_limits_violation =
    R"doc(True if the robot moved past the joint limits.)doc";

static const char *mkd_doc_franka_Errors_joint_position_motion_generator_start_pose_invalid =
    R"doc(True if an external joint position motion generator was started with a pose too far from the current
pose.)doc";

static const char *mkd_doc_franka_Errors_joint_reflex =
    R"doc(True if a collision was detected, i.e. the robot exceeded a torque threshold in a joint motion.)doc";

static const char *mkd_doc_franka_Errors_joint_velocity_violation =
    R"doc(True if the robot exceeded joint velocity limits.)doc";

static const char *mkd_doc_franka_Errors_joint_via_motion_generator_planning_joint_limit_violation =
    R"doc(True if the generated motion violates a joint limit.)doc";

static const char *mkd_doc_franka_Errors_max_goal_pose_deviation_violation =
    R"doc(True if internal motion generator did not reach the goal pose.)doc";

static const char *mkd_doc_franka_Errors_max_path_pose_deviation_violation =
    R"doc(True if internal motion generator deviated from the path.)doc";

static const char *mkd_doc_franka_Errors_operator_assign =
    R"doc(Assigns this Errors instance from another Errors value.

Args:
    other: Other Errors instance.

Returns:
    Errors instance.

)doc";

static const char *mkd_doc_franka_Errors_operator_basic_string =
    R"doc(Creates a string with names of active errors: "[active_error_name2, active_error_name_2, ...
active_error_name_n]" If no errors are active, the string contains empty brackets: "[]"

Returns:
    string with names of active errors

)doc";

static const char *mkd_doc_franka_Errors_operator_bool =
    R"doc(Check if any error flag is set to true.

Returns:
    True if any errors are set.

)doc";

static const char *mkd_doc_franka_Errors_power_limit_violation =
    R"doc(True if commanded values would result in exceeding the power limit.)doc";

static const char *mkd_doc_franka_Errors_self_collision_avoidance_violation =
    R"doc(True if the robot would have collided with itself.)doc";

static const char *mkd_doc_franka_Errors_start_elbow_sign_inconsistent =
    R"doc(True if the start elbow sign was inconsistent.

Applies only to motions started from Desk.)doc";

static const char *mkd_doc_franka_Errors_tau_j_range_violation =
    R"doc(True if the measured torque signal is out of the safe range.)doc";

static const char *mkd_doc_franka_Exception = R"doc(Base class for all exceptions used by `libfranka`.)doc";

static const char *mkd_doc_franka_Finishable =
    R"doc(Helper type for control and motion generation loops.

Used to determine whether to terminate a loop after the control callback has returned.

See also:
    callback-docs "Documentation on callbacks")doc";

static const char *mkd_doc_franka_Finishable_motion_finished =
    R"doc(Determines whether to finish a currently running motion.)doc";

static const char *mkd_doc_franka_Frame =
    R"doc(Enumerates the seven joints, the flange, and the end effector of a robot.)doc";

static const char *mkd_doc_franka_Frame_kEndEffector = R"doc()doc";

static const char *mkd_doc_franka_Frame_kFlange = R"doc()doc";

static const char *mkd_doc_franka_Frame_kJoint1 = R"doc()doc";

static const char *mkd_doc_franka_Frame_kJoint2 = R"doc()doc";

static const char *mkd_doc_franka_Frame_kJoint3 = R"doc()doc";

static const char *mkd_doc_franka_Frame_kJoint4 = R"doc()doc";

static const char *mkd_doc_franka_Frame_kJoint5 = R"doc()doc";

static const char *mkd_doc_franka_Frame_kJoint6 = R"doc()doc";

static const char *mkd_doc_franka_Frame_kJoint7 = R"doc()doc";

static const char *mkd_doc_franka_Frame_kStiffness = R"doc()doc";

static const char *mkd_doc_franka_Gripper =
    R"doc(Maintains a network connection to the gripper, provides the current gripper state, and allows the
execution of commands.

Note:
    The members of this class are threadsafe.)doc";

static const char *mkd_doc_franka_GripperState = R"doc(Describes the gripper state.)doc";

static const char *mkd_doc_franka_GripperState_is_grasped =
    R"doc(Indicates whether an object is currently grasped.)doc";

static const char *mkd_doc_franka_GripperState_max_width =
    R"doc(Maximum gripper opening width. This parameter is estimated by homing the gripper. After changing the
gripper fingers, a homing needs to be done. Unit: :math:`[m]`.

See also:
    Gripper::homing.)doc";

static const char *mkd_doc_franka_GripperState_temperature =
    R"doc(Current gripper temperature. Unit: :math:`[°C]`.)doc";

static const char *mkd_doc_franka_GripperState_time =
    R"doc(Strictly monotonically increasing timestamp since robot start.)doc";

static const char *mkd_doc_franka_GripperState_width = R"doc(Current gripper opening width. Unit: :math:`[m]`.)doc";

static const char *mkd_doc_franka_Gripper_Gripper =
    R"doc(Establishes a connection with a gripper connected to a robot.

Args:
    franka_address: IP/hostname of the robot the gripper is connected to.

Raises:
    NetworkException: if the connection is unsuccessful.
    IncompatibleVersionException: if this version of `libfranka` is not supported.

)doc";

static const char *mkd_doc_franka_Gripper_Gripper_2 =
    R"doc(Move-constructs a new Gripper instance.

Args:
    gripper: Other Gripper instance.

)doc";

static const char *mkd_doc_franka_Gripper_Gripper_3 = R"doc(@cond DO_NOT_DOCUMENT)doc";

static const char *mkd_doc_franka_Gripper_grasp =
    R"doc(Grasps an object.

An object is considered grasped if the distance :math:`d` between the gripper fingers satisfies
:math:`(\text{width} - \text{epsilon_inner}) < d < (\text{width} + \text{epsilon_outer})`.

Args:
    width: Size of the object to grasp in :math:`[m]`.
    speed: Closing speed in :math:`[\frac{m}{s}]`.
    force: Grasping force in :math:`[N]`.
    epsilon_inner: Maximum tolerated deviation when the actual grasped width is smaller than
                        the commanded grasp width.
    epsilon_outer: Maximum tolerated deviation when the actual grasped width is larger than the
                        commanded grasp width.

Returns:
    True if an object has been grasped, false otherwise.

Raises:
    CommandException: if an error occurred.
    NetworkException: if the connection is lost, e.g. after a timeout.

)doc";

static const char *mkd_doc_franka_Gripper_homing =
    R"doc(Performs homing of the gripper.

After changing the gripper fingers, a homing needs to be done. This is needed to estimate the
maximum grasping width.

Returns:
    True if command was successful, false otherwise.

Raises:
    CommandException: if an error occurred.
    NetworkException: if the connection is lost, e.g. after a timeout.

See also:
    GripperState for the maximum grasping width.

)doc";

static const char *mkd_doc_franka_Gripper_move =
    R"doc(Moves the gripper fingers to a specified width.

Args:
    width: Intended opening width in :math:`[m]`.
    speed: Closing speed in :math:`[\frac{m}{s}]`.

Returns:
    True if command was successful, false otherwise.

Raises:
    CommandException: if an error occurred.
    NetworkException: if the connection is lost, e.g. after a timeout.

)doc";

static const char *mkd_doc_franka_Gripper_network = R"doc(@endcond)doc";

static const char *mkd_doc_franka_Gripper_operator_assign =
    R"doc(Move-assigns this Gripper from another Gripper instance.

Args:
    gripper: Other Gripper instance.

Returns:
    Model instance.

)doc";

static const char *mkd_doc_franka_Gripper_operator_assign_2 = R"doc()doc";

static const char *mkd_doc_franka_Gripper_readOnce =
    R"doc(Waits for a gripper state update and returns it.

Returns:
    Current gripper state.

Raises:
    NetworkException: if the connection is lost, e.g. after a timeout.
    InvalidOperationException: if another readOnce is already running.

)doc";

static const char *mkd_doc_franka_Gripper_ri_version = R"doc()doc";

static const char *mkd_doc_franka_Gripper_serverVersion =
    R"doc(Returns the software version reported by the connected server.

Returns:
    Software version of the connected server.

)doc";

static const char *mkd_doc_franka_Gripper_stop =
    R"doc(Stops a currently running gripper move or grasp.

Returns:
    True if command was successful, false otherwise.

Raises:
    CommandException: if an error occurred.
    NetworkException: if the connection is lost, e.g. after a timeout.

)doc";

static const char *mkd_doc_franka_IncompatibleVersionException =
    R"doc(IncompatibleVersionException is thrown if the robot does not support this version of libfranka.)doc";

static const char *mkd_doc_franka_IncompatibleVersionException_IncompatibleVersionException =
    R"doc(Creates the exception using the two different protocol versions.

Args:
    server_version: Protocol version on the Control side.
    library_version: Protocol version of libfranka.

)doc";

static const char *mkd_doc_franka_IncompatibleVersionException_library_version = R"doc(libfranka protocol version.)doc";

static const char *mkd_doc_franka_IncompatibleVersionException_server_version = R"doc(Control's protocol version.)doc";

static const char *mkd_doc_franka_InvalidOperationException =
    R"doc(InvalidOperationException is thrown if an operation cannot be performed.)doc";

static const char *mkd_doc_franka_JointPositions = R"doc(Stores values for joint position motion generation.)doc";

static const char *mkd_doc_franka_JointPositions_JointPositions =
    R"doc(Creates a new JointPositions instance.

Args:
    joint_positions: Desired joint angles in :math:`[rad]`.

)doc";

static const char *mkd_doc_franka_JointPositions_JointPositions_2 =
    R"doc(Creates a new JointPositions instance.

Args:
    joint_positions: Desired joint angles in :math:`[rad]`.

Raises:
    std::invalid_argument: if the given initializer list has an invalid number of arguments.

)doc";

static const char *mkd_doc_franka_JointPositions_q = R"doc(Desired joint angles in [rad].)doc";

static const char *mkd_doc_franka_JointVelocities = R"doc(Stores values for joint velocity motion generation.)doc";

static const char *mkd_doc_franka_JointVelocities_JointVelocities =
    R"doc(Creates a new JointVelocities instance.

Args:
    joint_velocities: Desired joint velocities in :math:`[\frac{rad}{s}]`.

)doc";

static const char *mkd_doc_franka_JointVelocities_JointVelocities_2 =
    R"doc(Creates a new JointVelocities instance.

Args:
    joint_velocities: Desired joint velocities in :math:`[\frac{rad}{s}]`.

Raises:
    std::invalid_argument: if the given initializer list has an invalid number of arguments.

)doc";

static const char *mkd_doc_franka_JointVelocities_dq = R"doc(Desired joint velocities in :math:`[\frac{rad}{s}]`.)doc";

static const char *mkd_doc_franka_JointVelocityLimitsConfig =
    R"doc(Configuration class that holds position-based joint velocity limit parameters for all 7 joints. This
class handles parsing URDF data, storing joint configurations, and computing velocity limits.)doc";

static const char *mkd_doc_franka_JointVelocityLimitsConfig_JointVelocityLimitsConfig = R"doc()doc";

static const char *mkd_doc_franka_JointVelocityLimitsConfig_JointVelocityLimitsConfig_2 =
    R"doc(Constructor that parses URDF string and initializes joint parameters

Args:
    urdf_string: The URDF string to parse for joint velocity limit parameters

)doc";

static const char *mkd_doc_franka_JointVelocityLimitsConfig_getJointIndex =
    R"doc(Get joint index from joint name using efficient lookup

Args:
    joint_name: The name of the joint to find index for

Returns:
    Joint index (0-6) or -1 if not found

)doc";

static const char *mkd_doc_franka_JointVelocityLimitsConfig_getJointParams =
    R"doc(Get all joint parameters

Returns:
    Array of joint velocity limit parameters for all joints

)doc";

static const char *mkd_doc_franka_JointVelocityLimitsConfig_getLowerJointVelocityLimits =
    R"doc(Compute lower joint velocity limits based on current joint positions

Args:
    joint_positions: Current joint positions

Returns:
    Array of lower velocity limits for each joint

)doc";

static const char *mkd_doc_franka_JointVelocityLimitsConfig_getUpperJointVelocityLimits =
    R"doc(Compute upper joint velocity limits based on current joint positions

Args:
    joint_positions: Current joint positions

Returns:
    Array of upper velocity limits for each joint

)doc";

static const char *mkd_doc_franka_JointVelocityLimitsConfig_joint_params = R"doc()doc";

static const char *mkd_doc_franka_JointVelocityLimitsConfig_operator_array =
    R"doc(Get joint parameters for a specific joint

Args:
    joint_index: Index of the joint (0-6)

Returns:
    Joint velocity limit parameters for the specified joint

)doc";

static const char *mkd_doc_franka_JointVelocityLimitsConfig_parseFromURDF =
    R"doc(Parse joint velocity limit parameters from URDF string

Args:
    urdf_string: The URDF string to parse

)doc";

static const char *mkd_doc_franka_MobileFrame = R"doc(Enumerates the frames of a mobile robot's drive modules.)doc";

static const char *mkd_doc_franka_MobileFrame_kFrontDriveModule = R"doc(Front swerve drive module frame)doc";

static const char *mkd_doc_franka_MobileFrame_kRearDriveModule = R"doc(Rear swerve drive module frame)doc";

static const char *mkd_doc_franka_MobileJointPositions =
    R"doc(Joint positions for all controllable subsystems of a mobile robot.

All arrays are default-initialized to zero (neutral configuration).)doc";

static const char *mkd_doc_franka_MobileJointPositions_drive_modules =
    R"doc(Drive module joint positions: {front_steering, front_drive, rear_steering, rear_drive}.)doc";

static const char *mkd_doc_franka_MobileModel =
    R"doc(Calculates poses for mobile robot frames using Pinocchio.

This class provides forward kinematics for mobile robots. It is constructed from a mobile robot URDF
and will reject arm robot URDFs.

Note:
    The pose() method is RT-safe (zero heap allocation). Passive joints (casters, rocker arm) are
    held at their neutral configuration internally.)doc";

static const char *mkd_doc_franka_MobileModel_Data = R"doc()doc";

static const char *mkd_doc_franka_MobileModel_MobileModel =
    R"doc(Constructs a MobileModel from a URDF string.

Args:
    urdf_model: The URDF model string describing a mobile robot.

Raises:
    ModelException: if the URDF describes an arm robot rather than a mobile robot.

)doc";

static const char *mkd_doc_franka_MobileModel_MobileModel_2 =
    R"doc(Move-constructs a new MobileModel instance.

Args:
    other: Other MobileModel instance.

)doc";

static const char *mkd_doc_franka_MobileModel_MobileModel_3 = R"doc(@cond DO_NOT_DOCUMENT)doc";

static const char *mkd_doc_franka_MobileModel_data = R"doc()doc";

static const char *mkd_doc_franka_MobileModel_operator_assign =
    R"doc(Move-assigns this MobileModel from another MobileModel instance.

Args:
    other: Other MobileModel instance.

Returns:
    MobileModel instance.

)doc";

static const char *mkd_doc_franka_MobileModel_operator_assign_2 = R"doc()doc";

static const char *mkd_doc_franka_MobileModel_pose =
    R"doc(Gets the 4x4 pose matrix for the given mobile frame relative to the robot's base frame (URDF root
link, body-fixed).

The pose is represented as a 4x4 matrix in column-major format. This method is RT-safe: it performs
zero heap allocation.

Args:
    frame: The desired drive module frame (kFrontDriveModule or kRearDriveModule).
    joint_positions: Joint positions for all controllable subsystems. Angles are in radians;
                          the conversion to the internal Pinocchio representation is handled
                          automatically.

Returns:
    Vectorized 4x4 pose matrix, column-major.

)doc";

static const char *mkd_doc_franka_Model = R"doc(Calculates poses of joints and dynamic properties of the robot.)doc";

static const char *mkd_doc_franka_Model_2 = R"doc()doc";

static const char *mkd_doc_franka_ModelException =
    R"doc(ModelException is thrown if an error occurs when loading the model library.)doc";

static const char *mkd_doc_franka_Model_Model =
    R"doc(Creates a new Model instance.

This constructor is for internal use only.

Args:
    urdf_model: The URDF model string from the robot.

Raises:
    ModelException: if the model cannot be initialized.

See also:
    Robot::loadModel

)doc";

static const char *mkd_doc_franka_Model_Model_2 =
    R"doc(Creates a new Model instance only for the tests.

This constructor is for the unittests for enabling mocks.

Args:
    robot_model: unique pointer to the mocked robot_model

)doc";

static const char *mkd_doc_franka_Model_Model_3 =
    R"doc(Move-constructs a new Model instance.

Args:
    model: Other Model instance.

)doc";

static const char *mkd_doc_franka_Model_Model_4 = R"doc(@cond DO_NOT_DOCUMENT)doc";

static const char *mkd_doc_franka_Model_bodyJacobian =
    R"doc(Gets the 6x7 Jacobian for the given frame, relative to that frame.

The Jacobian is represented as a 6x7 matrix in column-major format.

Args:
    frame: The desired frame.
    robot_state: State from which the pose should be calculated.

Returns:
    Vectorized 6x7 Jacobian, column-major.

)doc";

static const char *mkd_doc_franka_Model_bodyJacobian_2 =
    R"doc(Gets the 6x7 Jacobian for the given frame, relative to that frame.

The Jacobian is represented as a 6x7 matrix in column-major format.

Args:
    frame: The desired frame.
    q: Joint position.
    F_T_EE: End effector in flange frame.
    EE_T_K: Stiffness frame K in the end effector frame.

Returns:
    Vectorized 6x7 Jacobian, column-major.

)doc";

static const char *mkd_doc_franka_Model_coriolis =
    R"doc(Calculates the Coriolis force vector (state-space equation): :math:` c= C \times dq`, in
:math:`[Nm]`.

Args:
    robot_state: State from which the Coriolis force vector should be calculated.

Returns:
    Coriolis force vector.

)doc";

static const char *mkd_doc_franka_Model_coriolis_2 =
    R"doc(Calculates the Coriolis force vector (state-space equation): :math:` c= C \times dq`, in
:math:`[Nm]`.

Args:
    q: Joint position.
    dq: Joint velocity.
    I_total: Inertia of the attached total load including end effector, relative to center of
                  mass, given as vectorized 3x3 column-major matrix. Unit: :math:`[kg \times m^2]`.
    m_total: Weight of the attached total load including end effector. Unit: :math:`[kg]`.
    F_x_Ctotal: Translation from flange to center of mass of the attached total load. Unit:
                     :math:`[m]`.

Returns:
    Coriolis force vector.

Deprecated:
    This overload is deprecated. Use coriolis(q, dq, i_total, m_total, f_x_ctotal, g_earth) for
    better performance.

)doc";

static const char *mkd_doc_franka_Model_coriolis_3 =
    R"doc( Calculates the Coriolis force vector with gravity (faster implementation).
 Unit: :math:`[Nm]`.

Args:
    q: Joint position.
    dq: Joint velocity.
    I_total: Inertia of the attached total load including end effector, relative to center of
                  mass, given as vectorized 3x3 column-major matrix. Unit: :math:`[kg \times m^2]`.
    m_total: Weight of the attached total load including end effector. Unit: :math:`[kg]`.
    F_x_Ctotal: Translation from flange to center of mass of the attached total load. Unit:
                     :math:`[m]`.
    gravity_earth: Earth's gravity vector. Unit: :math:`\frac{m}{s^2}`.

Returns:
    Coriolis force vector.

)doc";

static const char *mkd_doc_franka_Model_gravity =
    R"doc(Calculates the gravity vector. Unit: :math:`[Nm]`.

Args:
    q: Joint position.
    m_total: Weight of the attached total load including end effector. Unit: :math:`[kg]`.
    F_x_Ctotal: Translation from flange to center of mass of the attached total load. Unit:
                     :math:`[m]`.
    gravity_earth: Earth's gravity vector. Unit: :math:`\frac{m}{s^2}`. Default to {0.0, 0.0,
                        -9.81}.

Returns:
    Gravity vector.

)doc";

static const char *mkd_doc_franka_Model_gravity_2 =
    R"doc(Calculates the gravity vector. Unit: :math:`[Nm]`.

Args:
    robot_state: State from which the gravity vector should be calculated.
    gravity_earth: Earth's gravity vector. Unit: :math:`\frac{m}{s^2}`.

Returns:
    Gravity vector.

)doc";

static const char *mkd_doc_franka_Model_gravity_3 =
    R"doc(Calculates the gravity vector using the robot state. Unit: :math:`[Nm]`.

Args:
    robot_state: State from which the gravity vector should be calculated.

Returns:
    Gravity vector.

)doc";

static const char *mkd_doc_franka_Model_mass =
    R"doc(Calculates the 7x7 mass matrix. Unit: :math:`[kg \times m^2]`.

Args:
    robot_state: State from which the mass matrix should be calculated.

Returns:
    Vectorized 7x7 mass matrix, column-major.

)doc";

static const char *mkd_doc_franka_Model_mass_2 =
    R"doc(Calculates the 7x7 mass matrix. Unit: :math:`[kg \times m^2]`.

Args:
    q: Joint position.
    I_total: Inertia of the attached total load including end effector, relative to center of
                  mass, given as vectorized 3x3 column-major matrix. Unit: :math:`[kg \times m^2]`.
    m_total: Weight of the attached total load including end effector. Unit: :math:`[kg]`.
    F_x_Ctotal: Translation from flange to center of mass of the attached total load. Unit:
                     :math:`[m]`.

Returns:
    Vectorized 7x7 mass matrix, column-major.

)doc";

static const char *mkd_doc_franka_Model_operator_assign =
    R"doc(Move-assigns this Model from another Model instance.

Args:
    model: Other Model instance.

Returns:
    Model instance.

)doc";

static const char *mkd_doc_franka_Model_operator_assign_2 = R"doc()doc";

static const char *mkd_doc_franka_Model_pose =
    R"doc(Gets the 4x4 pose matrix for the given frame in base frame.

The pose is represented as a 4x4 matrix in column-major format.

Args:
    frame: The desired frame.
    robot_state: State from which the pose should be calculated.

Returns:
    Vectorized 4x4 pose matrix, column-major.

)doc";

static const char *mkd_doc_franka_Model_pose_2 =
    R"doc(Gets the 4x4 pose matrix for the given frame in base frame.

The pose is represented as a 4x4 matrix in column-major format.

Args:
    frame: The desired frame.
    q: Joint position.
    F_T_EE: End effector in flange frame.
    EE_T_K: Stiffness frame K in the end effector frame.

Returns:
    Vectorized 4x4 pose matrix, column-major.

)doc";

static const char *mkd_doc_franka_Model_robot_model = R"doc(@endcond)doc";

static const char *mkd_doc_franka_Model_zeroJacobian =
    R"doc(Gets the 6x7 Jacobian for the given joint relative to the base frame.

The Jacobian is represented as a 6x7 matrix in column-major format.

Args:
    frame: The desired frame.
    robot_state: State from which the pose should be calculated.

Returns:
    Vectorized 6x7 Jacobian, column-major.

)doc";

static const char *mkd_doc_franka_Model_zeroJacobian_2 =
    R"doc(Gets the 6x7 Jacobian for the given joint relative to the base frame.

The Jacobian is represented as a 6x7 matrix in column-major format.

Args:
    frame: The desired frame.
    q: Joint position.
    F_T_EE: End effector in flange frame.
    EE_T_K: Stiffness frame K in the end effector frame.

Returns:
    Vectorized 6x7 Jacobian, column-major.

)doc";

static const char *mkd_doc_franka_MotionFinished =
    R"doc(Helper method to indicate that a motion should stop after processing the given command.

Args:
    command: Last command to be executed before the motion terminates.

Returns:
    Command with motion_finished set to true.

See also:
    callback-docs "Documentation on callbacks"

)doc";

static const char *mkd_doc_franka_MotionFinished_2 =
    R"doc(Helper method to indicate that a motion should stop after processing the given command.

Args:
    command: Last command to be executed before the motion terminates.

Returns:
    Command with motion_finished set to true.

See also:
    callback-docs "Documentation on callbacks"

)doc";

static const char *mkd_doc_franka_MotionFinished_3 =
    R"doc(Helper method to indicate that a motion should stop after processing the given command.

Args:
    command: Last command to be executed before the motion terminates.

Returns:
    Command with motion_finished set to true.

See also:
    callback-docs "Documentation on callbacks"

)doc";

static const char *mkd_doc_franka_MotionFinished_4 =
    R"doc(Helper method to indicate that a motion should stop after processing the given command.

Args:
    command: Last command to be executed before the motion terminates.

Returns:
    Command with motion_finished set to true.

See also:
    callback-docs "Documentation on callbacks"

)doc";

static const char *mkd_doc_franka_MotionFinished_5 =
    R"doc(Helper method to indicate that a motion should stop after processing the given command.

Args:
    command: Last command to be executed before the motion terminates.

Returns:
    Command with motion_finished set to true.

See also:
    callback-docs "Documentation on callbacks"

)doc";

static const char *mkd_doc_franka_Network = R"doc()doc";

static const char *mkd_doc_franka_Network_2 = R"doc()doc";

static const char *mkd_doc_franka_NetworkException =
    R"doc(NetworkException is thrown if a connection to the robot cannot be established, or when a timeout
occurs.)doc";

static const char *mkd_doc_franka_PositionBasedJointVelocityLimitConstants =
    R"doc(Structure to hold position-based joint velocity limit constants for a single joint from URDF. These
parameters define velocity limits that depend on the current joint position.)doc";

static const char *mkd_doc_franka_PositionBasedJointVelocityLimitConstants_PositionBasedJointVelocityLimitConstants =
    R"doc()doc";

static const char *mkd_doc_franka_PositionBasedJointVelocityLimitConstants_PositionBasedJointVelocityLimitConstants_2 =
    R"doc(Constructor with specified parameters

Args:
    max_velocity: Maximum velocity
    velocity_offset: Velocity offset parameter
    deceleration_limit: Deceleration limit parameter
    upper_position_limit: Upper position limit
    lower_position_limit: Lower position limit

)doc";

static const char *mkd_doc_franka_PositionBasedJointVelocityLimitConstants_deceleration_limit = R"doc()doc";

static const char *mkd_doc_franka_PositionBasedJointVelocityLimitConstants_lower_position_limit = R"doc()doc";

static const char *mkd_doc_franka_PositionBasedJointVelocityLimitConstants_max_velocity = R"doc()doc";

static const char *mkd_doc_franka_PositionBasedJointVelocityLimitConstants_upper_position_limit = R"doc()doc";

static const char *mkd_doc_franka_PositionBasedJointVelocityLimitConstants_velocity_offset = R"doc()doc";

static const char *mkd_doc_franka_ProtocolException =
    R"doc(ProtocolException is thrown if the robot returns an incorrect message.)doc";

static const char *mkd_doc_franka_RealtimeConfig =
    R"doc(Used to decide whether to enforce realtime mode for a control loop thread.

See also:
    Robot::Robot)doc";

static const char *mkd_doc_franka_RealtimeConfig_kEnforce = R"doc()doc";

static const char *mkd_doc_franka_RealtimeConfig_kIgnore = R"doc()doc";

static const char *mkd_doc_franka_RealtimeException =
    R"doc(RealtimeException is thrown if realtime priority cannot be set.)doc";

static const char *mkd_doc_franka_Robot =
    R"doc(Maintains a network connection to the robot, provides the current robot state, gives access to the
model library and allows to control the robot.


@anchor o-frame

@anchor f-frame

@anchor ne-frame

@anchor ee-frame

@anchor k-frame

Note:
    The members of this class are threadsafe.

Base frame O:
    The base frame is located at the center of the robot's base. Its z-axis is identical with the
    axis of rotation of the first joint.

Flange frame F:
    The flange frame is located at the center of the flange surface. Its z-axis is identical with
    the axis of rotation of the last joint. This frame is fixed and cannot be changed.

Nominal end effector frame NE:
    The nominal end effector frame is configured outside of libfranka (in DESK) and cannot be
    changed here. It may be used to set end effector frames which are rarely changed.

end effector frame EE:
    By default, the end effector frame EE is the same as the nominal end effector frame NE (i.e. the
    transformation between NE and EE is the identity transformation). It may be used to set end
    effector frames which are changed more frequently (such as a tool that is grasped with the end
    effector). With Robot::setEE, a custom transformation matrix can be set.

Stiffness frame K:
    This frame describes the transformation from the end effector frame EE to the stiffness frame K.
    The stiffness frame is used for Cartesian impedance control, and for measuring and applying
    forces. The values set using Robot::setCartesianImpedance are used in the direction of the
    stiffness frame. It can be set with Robot::setK. This frame allows to modify the compliance
    behavior of the robot (e.g. to have a low stiffness around a specific point which is not the end
    effector). The stiffness frame does not affect where the robot will move to. The user should
    always command the desired end effector frame (and not the desired stiffness frame).)doc";

static const char *mkd_doc_franka_RobotMode = R"doc(Describes the robot's current mode.)doc";

static const char *mkd_doc_franka_RobotMode_kAutomaticErrorRecovery = R"doc()doc";

static const char *mkd_doc_franka_RobotMode_kGuiding = R"doc()doc";

static const char *mkd_doc_franka_RobotMode_kIdle = R"doc()doc";

static const char *mkd_doc_franka_RobotMode_kMove = R"doc()doc";

static const char *mkd_doc_franka_RobotMode_kOther = R"doc()doc";

static const char *mkd_doc_franka_RobotMode_kReflex = R"doc()doc";

static const char *mkd_doc_franka_RobotMode_kUserStopped = R"doc()doc";

static const char *mkd_doc_franka_RobotModelBase = R"doc()doc";

static const char *mkd_doc_franka_RobotModelBase_2 = R"doc()doc";

static const char *mkd_doc_franka_RobotState = R"doc(Describes the robot state.)doc";

static const char *mkd_doc_franka_RobotState_EE_T_K =
    R"doc(:math:`^{EE}T_{K}` Stiffness frame pose in end effector frame. Pose is represented as a 4x4 matrix
in column-major format.

See also k-frame "K frame".)doc";

static const char *mkd_doc_franka_RobotState_F_T_EE =
    R"doc(:math:`^{F}T_{EE}` End effector frame pose in flange frame. Pose is represented as a 4x4 matrix in
column-major format.

See also:
    F_T_NE

See also:
    NE_T_EE

See also:
    Robot for an explanation of the F, NE and EE frames.)doc";

static const char *mkd_doc_franka_RobotState_F_T_NE =
    R"doc(:math:`^{F}T_{NE}` Nominal end effector frame pose in flange frame. Pose is represented as a 4x4
matrix in column-major format.

See also:
    F_T_EE

See also:
    NE_T_EE

See also:
    Robot for an explanation of the F, NE and EE frames.)doc";

static const char *mkd_doc_franka_RobotState_F_x_Cee =
    R"doc(:math:`^{F}x_{C_{EE}}` Configured center of mass of the end effector load with respect to flange
frame.)doc";

static const char *mkd_doc_franka_RobotState_F_x_Cload =
    R"doc(:math:`^{F}x_{C_{load}}` Configured center of mass of the external load with respect to flange
frame.)doc";

static const char *mkd_doc_franka_RobotState_F_x_Ctotal =
    R"doc(:math:`^{F}x_{C_{total}}` Combined center of mass of the end effector load and the external load
with respect to flange frame.)doc";

static const char *mkd_doc_franka_RobotState_I_ee =
    R"doc(:math:`I_{EE}` Configured rotational inertia matrix of the end effector load with respect to center
of mass.)doc";

static const char *mkd_doc_franka_RobotState_I_load =
    R"doc(:math:`I_{load}` Configured rotational inertia matrix of the external load with respect to center of
mass.)doc";

static const char *mkd_doc_franka_RobotState_I_total =
    R"doc(:math:`I_{total}` Combined rotational inertia matrix of the end effector load and the external load
with respect to the center of mass.)doc";

static const char *mkd_doc_franka_RobotState_K_F_ext_hat_K =
    R"doc(:math:`^{K}F_{K,\text{ext}}` Estimated external wrench (force, torque) acting on stiffness frame,
expressed relative to the stiffness frame. Forces applied by the robot to the environment are
positive, while forces applied by the environment on the robot are negative. Becomes
:math:`[0,0,0,0,0,0]` when near or in a singularity. See also k-frame "Stiffness frame K".
Unit: :math:`[N,N,N,Nm,Nm,Nm]`.)doc";

static const char *mkd_doc_franka_RobotState_NE_T_EE =
    R"doc(:math:`^{NE}T_{EE}` End effector frame pose in nominal end effector frame. Pose is represented as a
4x4 matrix in column-major format.

See also:
    Robot::setEE to change this frame.

See also:
    F_T_EE

See also:
    F_T_NE

See also:
    Robot for an explanation of the F, NE and EE frames.)doc";

static const char *mkd_doc_franka_RobotState_O_F_ext_hat_K =
    R"doc(:math:`^OF_{K,\text{ext}}` Estimated external wrench (force, torque) acting on stiffness frame,
expressed relative to the o-frame "base frame". Forces applied by the robot to the environment are
positive, while forces applied by the environment on the robot are negative. Becomes
:math:`[0,0,0,0,0,0]` when near or in a singularity. See also k-frame "Stiffness frame K".
Unit: :math:`[N,N,N,Nm,Nm,Nm]`.)doc";

static const char *mkd_doc_franka_RobotState_O_T_EE =
    R"doc(:math:`^{O}T_{EE}` Measured end effector pose in o-frame "base frame". Pose is represented as a 4x4
matrix in column-major format.)doc";

static const char *mkd_doc_franka_RobotState_O_T_EE_c =
    R"doc(:math:`{^OT_{EE}}_{c}` Last commanded end effector pose of motion generation in o-frame "base
frame". Pose is represented as a 4x4 matrix in column-major format.)doc";

static const char *mkd_doc_franka_RobotState_O_T_EE_d =
    R"doc(:math:`{^OT_{EE}}_{d}` Last desired end effector pose of motion generation in o-frame "base frame".
Pose is represented as a 4x4 matrix in column-major format.)doc";

static const char *mkd_doc_franka_RobotState_O_dP_EE_c =
    R"doc(:math:`{^OdP_{EE}}_{c}` Last commanded end effector twist in o-frame "base frame".
Unit: :math:`[\frac{m}{s},\frac{m}{s},\frac{m}{s},\frac{rad}{s},\frac{rad}{s},\frac{rad}{s}]`.)doc";

static const char *mkd_doc_franka_RobotState_O_dP_EE_d =
    R"doc(:math:`{^OdP_{EE}}_{d}` Desired end effector twist in o-frame "base frame".
Unit: :math:`[\frac{m}{s},\frac{m}{s},\frac{m}{s},\frac{rad}{s},\frac{rad}{s},\frac{rad}{s}]`.)doc";

static const char *mkd_doc_franka_RobotState_O_ddP_EE_c =
    R"doc(:math:`{^OddP_{EE}}_{c}` Last commanded end effector acceleration in o-frame "base frame".
Unit:
:math:`[\frac{m}{s^2},\frac{m}{s^2},\frac{m}{s^2},\frac{rad}{s^2},\frac{rad}{s^2},\frac{rad}{s^2}]`.)doc";

static const char *mkd_doc_franka_RobotState_O_ddP_O =
    R"doc(:math:`{^OddP}_O` Linear component of the acceleration of the robot's base, expressed in frame
parallel to the o-frame "base frame", i.e. the base's translational acceleration. If the base is
resting this shows the direction of the gravity vector. It is harcoded for now to `{0, 0, -9.81}`.)doc";

static const char *mkd_doc_franka_RobotState_accelerometer_bottom =
    R"doc(:math:`\ddot{x}` Translational acceleration data from the accelerometers located on the bottom of
the PCB of Joint 1 through Joint 6 (no accelerometers in the last joint). Unit:
:math:`[\frac{m}{s^2}]`)doc";

static const char *mkd_doc_franka_RobotState_accelerometer_top =
    R"doc(:math:`\ddot{x}` Translational acceleration data from the accelerometers located on the top of the
PCB of Joint 1 through Joint 6 (no accelerometers in the last joint). Unit: :math:`[\frac{m}{s^2}]`)doc";

static const char *mkd_doc_franka_RobotState_cartesian_collision =
    R"doc(Indicates which contact level is activated in which Cartesian dimension :math:`(x,y,z,R,P,Y)`. After
contact disappears, the value stays the same until a reset command is sent.

See also:
    Robot::setCollisionBehavior for setting sensitivity values.

See also:
    Robot::automaticErrorRecovery for performing a reset after a collision.)doc";

static const char *mkd_doc_franka_RobotState_cartesian_contact =
    R"doc(Indicates which contact level is activated in which Cartesian dimension :math:`(x,y,z,R,P,Y)`. After
contact disappears, the value turns to zero.

See also:
    Robot::setCollisionBehavior for setting sensitivity values.)doc";

static const char *mkd_doc_franka_RobotState_control_command_success_rate =
    R"doc(Percentage of the last 100 control commands that were successfully received by the robot.

Shows a value of zero if no control or motion generator loop is currently running.

Range: :math:`[0, 1]`.)doc";

static const char *mkd_doc_franka_RobotState_current_errors = R"doc(Current error state.)doc";

static const char *mkd_doc_franka_RobotState_ddelbow_c =
    R"doc(Commanded elbow acceleration.

The values of the array are:
 - ddelbow_c[0] Acceleration of the 3rd joint in :math:`\frac{rad}{s^2}` - ddelbow_c[1] is always 0.)doc";

static const char *mkd_doc_franka_RobotState_ddq_d =
    R"doc(:math:`\ddot{q}_d` Desired joint acceleration. Unit: :math:`[\frac{rad}{s^2}]`)doc";

static const char *mkd_doc_franka_RobotState_delbow_c =
    R"doc(Commanded elbow velocity.

The values of the array are:
 - delbow_c[0] Velocity of the 3rd joint in :math:`\frac{rad}{s}` - delbow_c[1] is always 0.)doc";

static const char *mkd_doc_franka_RobotState_dq =
    R"doc(:math:`\dot{q}` Measured joint velocity. Unit: :math:`[\frac{rad}{s}]`)doc";

static const char *mkd_doc_franka_RobotState_dq_d =
    R"doc(:math:`\dot{q}_d` Desired joint velocity. Unit: :math:`[\frac{rad}{s}]`)doc";

static const char *mkd_doc_franka_RobotState_dtau_J =
    R"doc(:math:`\dot{\tau_{J}}` Derivative of measured link-side joint torque sensor signals. Unit:
:math:`[\frac{Nm}{s}]`)doc";

static const char *mkd_doc_franka_RobotState_dtheta =
    R"doc(:math:`\dot{\theta}` Motor velocity. Unit: :math:`[\frac{rad}{s}]`)doc";

static const char *mkd_doc_franka_RobotState_elbow =
    R"doc(Elbow configuration.

The values of the array are:
 - elbow[0]: Position of the 3rd joint in :math:`[rad]`. - elbow[1]: Flip direction of the elbow
   (4th joint):
   - +1 if :math:`q_4 > q_{elbow-flip}` - 0 if :math:`q_4 == q_{elbow-flip} ` - -1 if :math:`q_4 <
     q_{elbow-flip} `
   . with :math:`q_{elbow-flip}` as specified in the robot interface specification page in the FCI
Documentation.)doc";

static const char *mkd_doc_franka_RobotState_elbow_c =
    R"doc(Commanded elbow configuration.

The values of the array are:
 - elbow_c[0]: Position of the 3rd joint in :math:`[rad]`. - elbow_c[1]: Flip direction of the elbow
   (4th joint):
   - +1 if :math:`q_4 > q_{elbow-flip}` - 0 if :math:`q_4 == q_{elbow-flip} ` - -1 if :math:`q_4 <
     q_{elbow-flip} `
   . with :math:`q_{elbow-flip}` as specified in the robot interface specification page in the FCI
Documentation.)doc";

static const char *mkd_doc_franka_RobotState_elbow_d =
    R"doc(Desired elbow configuration.

The values of the array are:
 - elbow_d[0]: Position of the 3rd joint in :math:`[rad]`. - elbow_d[1]: Flip direction of the elbow
   (4th joint):
   - +1 if :math:`q_4 > q_{elbow-flip}` - 0 if :math:`q_4 == q_{elbow-flip} ` - -1 if :math:`q_4 <
     q_{elbow-flip} `
   . with :math:`q_{elbow-flip}` as specified in the robot interface specification page in the FCI
Documentation.)doc";

static const char *mkd_doc_franka_RobotState_joint_collision =
    R"doc(Indicates which contact level is activated in which joint. After contact disappears, the value stays
the same until a reset command is sent.

See also:
    Robot::setCollisionBehavior for setting sensitivity values.

See also:
    Robot::automaticErrorRecovery for performing a reset after a collision.)doc";

static const char *mkd_doc_franka_RobotState_joint_contact =
    R"doc(Indicates which contact level is activated in which joint. After contact disappears, value turns to
zero.

See also:
    Robot::setCollisionBehavior for setting sensitivity values.)doc";

static const char *mkd_doc_franka_RobotState_last_motion_errors =
    R"doc(Contains the errors that aborted the previous motion.)doc";

static const char *mkd_doc_franka_RobotState_m_ee = R"doc(:math:`m_{EE}` Configured mass of the end effector.)doc";

static const char *mkd_doc_franka_RobotState_m_load = R"doc(:math:`m_{load}` Configured mass of the external load.)doc";

static const char *mkd_doc_franka_RobotState_m_total =
    R"doc(:math:`m_{total}` Sum of the mass of the end effector and the external load.)doc";

static const char *mkd_doc_franka_RobotState_q = R"doc(:math:`q` Measured joint position. Unit: :math:`[rad]`)doc";

static const char *mkd_doc_franka_RobotState_q_d = R"doc(:math:`q_d` Desired joint position. Unit: :math:`[rad]`)doc";

static const char *mkd_doc_franka_RobotState_robot_mode = R"doc(Current robot mode.)doc";

static const char *mkd_doc_franka_RobotState_tau_J =
    R"doc(:math:`\tau_{J}` Measured link-side joint torque sensor signals. Unit: :math:`[Nm]`)doc";

static const char *mkd_doc_franka_RobotState_tau_J_d =
    R"doc(:math:`{\tau_J}_d` Desired link-side joint torque sensor signals without gravity. Unit: :math:`[Nm]`)doc";

static const char *mkd_doc_franka_RobotState_tau_ext_hat_filtered =
    R"doc(:math:`\hat{\tau}_{\text{ext}}` Low-pass filtered torques generated by external forces on the
joints. It does not include configured end-effector and load nor the mass and dynamics of the robot.
tau_ext_hat_filtered is the error between tau_J and the expected torques given by the robot model.
Unit: :math:`[Nm]`.)doc";

static const char *mkd_doc_franka_RobotState_theta = R"doc(:math:`\theta` Motor position. Unit: :math:`[rad]`)doc";

static const char *mkd_doc_franka_RobotState_time =
    R"doc(Strictly monotonically increasing timestamp since robot start.

Inside of control loops callback-docs "time_step" parameter of Robot::control can be used instead.)doc";

static const char *mkd_doc_franka_Robot_Impl = R"doc(@endcond)doc";

static const char *mkd_doc_franka_Robot_Robot =
    R"doc(Establishes a connection with the robot.

Args:
    franka_address: IP/hostname of the robot.
    realtime_config: if set to Enforce, an exception will be thrown if realtime priority cannot
                          be set when required. Setting realtime_config to Ignore disables this
                          behavior.
    log_size: sets how many last states should be kept for logging purposes. The log is
                   provided when a ControlException is thrown.

Raises:
    NetworkException: if the connection is unsuccessful.
    IncompatibleVersionException: if this version of `libfranka` is not supported.

)doc";

static const char *mkd_doc_franka_Robot_Robot_2 =
    R"doc(Move-constructs a new Robot instance.

Args:
    other: Other Robot instance.

)doc";

static const char *mkd_doc_franka_Robot_Robot_3 = R"doc(@cond DO_NOT_DOCUMENT)doc";

static const char *mkd_doc_franka_Robot_Robot_4 =
    R"doc(Constructs a new Robot given a Robot::Impl. This enables unittests with Robot::Impl-Mocks.

Args:
    robot_impl: Robot::Impl to use

)doc";

static const char *mkd_doc_franka_Robot_Robot_5 = R"doc(Default constructor to enable mocking and testing.)doc";

static const char *mkd_doc_franka_Robot_automaticErrorRecovery =
    R"doc(Runs automatic error recovery on the robot.

Automatic error recovery e.g. resets the robot after a collision occurred.

Raises:
    CommandException: if the Control reports an error.
    NetworkException: if the connection is lost, e.g. after a timeout.

)doc";

static const char *mkd_doc_franka_Robot_control =
    R"doc(Starts a control loop for sending joint-level torque commands.

Sets realtime priority for the current thread. Cannot be executed while another control or motion
generator loop is active.

Args:
    control_callback: Callback function providing joint-level torque commands. See callback-
                           docs "here" for more details.
    limit_rate: True if rate limiting should be activated. False by default. This could distort
                     your motion!
    cutoff_frequency: Cutoff frequency for a first order low-pass filter applied on the user
                           commanded signal. Set to franka::kMaxCutoffFrequency to disable.

Raises:
    ControlException: if an error related to torque control or motion generation occurred.
    InvalidOperationException: if a conflicting operation is already running.
    NetworkException: if the connection is lost, e.g. after a timeout.
    RealtimeException: if realtime priority cannot be set for the current thread.
    std::invalid_argument: if joint-level torque commands are NaN or infinity.

See also:
    Robot::Robot to change behavior if realtime priority cannot be set.

)doc";

static const char *mkd_doc_franka_Robot_control_2 =
    R"doc(Starts a control loop for sending joint-level torque commands and joint positions.

Sets realtime priority for the current thread. Cannot be executed while another control or motion
generator loop is active.

Args:
    control_callback: Callback function providing joint-level torque commands. See callback-
                           docs "here" for more details.
    motion_generator_callback: Callback function for motion generation. See callback-docs
                                    "here" for more details.
    limit_rate: True if rate limiting should be activated. False by default. This could distort
                     your motion!
    cutoff_frequency: Cutoff frequency for a first order low-pass filter applied on the user
                           commanded signal. Set to franka::kMaxCutoffFrequency to disable.

Raises:
    ControlException: if an error related to torque control or motion generation occurred.
    InvalidOperationException: if a conflicting operation is already running.
    NetworkException: if the connection is lost, e.g. after a timeout.
    RealtimeException: if realtime priority cannot be set for the current thread.
    std::invalid_argument: if joint-level torque or joint position commands are NaN or infinity.

See also:
    Robot::Robot to change behavior if realtime priority cannot be set.

)doc";

static const char *mkd_doc_franka_Robot_control_3 =
    R"doc(Starts a control loop for sending joint-level torque commands and joint velocities.

Sets realtime priority for the current thread. Cannot be executed while another control or motion
generator loop is active.

Args:
    control_callback: Callback function providing joint-level torque commands. See callback-
                           docs "here" for more details.
    motion_generator_callback: Callback function for motion generation. See callback-docs
                                    "here" for more details.
    limit_rate: True if rate limiting should be activated. False by default. This could distort
                     your motion!
    cutoff_frequency: Cutoff frequency for a first order low-pass filter applied on the user
                           commanded signal. Set to franka::kMaxCutoffFrequency to disable.

Raises:
    ControlException: if an error related to torque control or motion generation occurred.
    InvalidOperationException: if a conflicting operation is already running.
    NetworkException: if the connection is lost, e.g. after a timeout.
    RealtimeException: if realtime priority cannot be set for the current thread.
    std::invalid_argument: if joint-level torque or joint velocity commands are NaN or infinity.

See also:
    Robot::Robot to change behavior if realtime priority cannot be set.

)doc";

static const char *mkd_doc_franka_Robot_control_4 =
    R"doc(Starts a control loop for sending joint-level torque commands and Cartesian poses.

Sets realtime priority for the current thread. Cannot be executed while another control or motion
generator loop is active.

Args:
    control_callback: Callback function providing joint-level torque commands. See callback-
                           docs "here" for more details.
    motion_generator_callback: Callback function for motion generation. See callback-docs
                                    "here" for more details.
    limit_rate: True if rate limiting should be activated. False by default. This could distort
                     your motion!
    cutoff_frequency: Cutoff frequency for a first order low-pass filter applied on the user
                           commanded signal. Set to franka::kMaxCutoffFrequency to disable.

Raises:
    ControlException: if an error related to torque control or motion generation occurred.
    InvalidOperationException: if a conflicting operation is already running.
    NetworkException: if the connection is lost, e.g. after a timeout.
    RealtimeException: if realtime priority cannot be set for the current thread.
    std::invalid_argument: if joint-level torque or Cartesian pose command elements are NaN or
                           infinity.

See also:
    Robot::Robot to change behavior if realtime priority cannot be set.

)doc";

static const char *mkd_doc_franka_Robot_control_5 =
    R"doc(Starts a control loop for sending joint-level torque commands and Cartesian velocities.

Sets realtime priority for the current thread. Cannot be executed while another control or motion
generator loop is active.

Args:
    control_callback: Callback function providing joint-level torque commands. See callback-
                           docs "here" for more details.
    motion_generator_callback: Callback function for motion generation. See callback-docs
                                    "here" for more details.
    limit_rate: True if rate limiting should be activated. False by default. This could distort
                     your motion!
    cutoff_frequency: Cutoff frequency for a first order low-pass filter applied on the user
                           commanded signal. Set to franka::kMaxCutoffFrequency to disable.

Raises:
    ControlException: if an error related to torque control or motion generation occurred.
    InvalidOperationException: if a conflicting operation is already running.
    NetworkException: if the connection is lost, e.g. after a timeout.
    RealtimeException: if realtime priority cannot be set for the current thread.
    std::invalid_argument: if joint-level torque or Cartesian velocity command elements are NaN or
                           infinity.

See also:
    Robot::Robot to change behavior if realtime priority cannot be set.

)doc";

static const char *mkd_doc_franka_Robot_control_6 =
    R"doc(Starts a control loop for a joint position motion generator with a given controller mode.

Sets realtime priority for the current thread. Cannot be executed while another control or motion
generator loop is active.

Args:
    motion_generator_callback: Callback function for motion generation. See callback-docs
                                    "here" for more details.
    controller_mode: Controller to use to execute the motion.
    limit_rate: True if rate limiting should be activated. False by default. This could distort
                     your motion!
    cutoff_frequency: Cutoff frequency for a first order low-pass filter applied on the user
                           commanded signal. Set to franka::kMaxCutoffFrequency to disable.

Raises:
    ControlException: if an error related to motion generation occurred.
    InvalidOperationException: if a conflicting operation is already running.
    NetworkException: if the connection is lost, e.g. after a timeout.
    RealtimeException: if realtime priority cannot be set for the current thread.
    std::invalid_argument: if joint position commands are NaN or infinity.

See also:
    Robot::Robot to change behavior if realtime priority cannot be set.

)doc";

static const char *mkd_doc_franka_Robot_control_7 =
    R"doc(Starts a control loop for a joint velocity motion generator with a given controller mode.

Sets realtime priority for the current thread. Cannot be executed while another control or motion
generator loop is active.

Args:
    motion_generator_callback: Callback function for motion generation. See callback-docs
                                    "here" for more details.
    controller_mode: Controller to use to execute the motion.
    limit_rate: True if rate limiting should be activated. False by default. This could distort
                     your motion!
    cutoff_frequency: Cutoff frequency for a first order low-pass filter applied on the user
                           commanded signal. Set to franka::kMaxCutoffFrequency to disable.

Raises:
    ControlException: if an error related to motion generation occurred.
    InvalidOperationException: if a conflicting operation is already running.
    NetworkException: if the connection is lost, e.g. after a timeout.
    RealtimeException: if realtime priority cannot be set for the current thread.
    std::invalid_argument: if joint velocity commands are NaN or infinity.

See also:
    Robot::Robot to change behavior if realtime priority cannot be set.

)doc";

static const char *mkd_doc_franka_Robot_control_8 =
    R"doc(Starts a control loop for a Cartesian pose motion generator with a given controller mode.

Sets realtime priority for the current thread. Cannot be executed while another control or motion
generator loop is active.

Args:
    motion_generator_callback: Callback function for motion generation. See callback-docs
                                    "here" for more details.
    controller_mode: Controller to use to execute the motion.
    limit_rate: True if rate limiting should be activated. False by default. This could distort
                     your motion!
    cutoff_frequency: Cutoff frequency for a first order low-pass filter applied on the user
                           commanded signal. Set to franka::kMaxCutoffFrequency to disable.

Raises:
    ControlException: if an error related to motion generation occurred.
    InvalidOperationException: if a conflicting operation is already running.
    NetworkException: if the connection is lost, e.g. after a timeout.
    RealtimeException: if realtime priority cannot be set for the current thread.
    std::invalid_argument: if Cartesian pose command elements are NaN or infinity.

See also:
    Robot::Robot to change behavior if realtime priority cannot be set.

)doc";

static const char *mkd_doc_franka_Robot_control_9 =
    R"doc(Starts a control loop for a Cartesian velocity motion generator with a given controller mode.

Sets realtime priority for the current thread. Cannot be executed while another control or motion
generator loop is active.

Args:
    motion_generator_callback: Callback function for motion generation. See callback-docs
                                    "here" for more details.
    controller_mode: Controller to use to execute the motion.
    limit_rate: True if rate limiting should be activated. False by default. This could distort
                     your motion!
    cutoff_frequency: Cutoff frequency for a first order low-pass filter applied on the user
                           commanded signal. Set to franka::kMaxCutoffFrequency to disable.

Raises:
    ControlException: if an error related to motion generation occurred.
    InvalidOperationException: if a conflicting operation is already running.
    NetworkException: if the connection is lost, e.g. after a timeout.
    RealtimeException: if realtime priority cannot be set for the current thread.
    std::invalid_argument: if Cartesian velocity command elements are NaN or infinity.

See also:
    Robot::Robot to change behavior if realtime priority cannot be set.

)doc";

static const char *mkd_doc_franka_Robot_control_mutex = R"doc()doc";

static const char *mkd_doc_franka_Robot_getLowerJointVelocityLimits =
    R"doc(Computes lower limits for joint velocities based on current joint positions.

Args:
    joint_positions: Joint positions.

Returns:
    Lower limits for joint velocities.

)doc";

static const char *mkd_doc_franka_Robot_getRobotModel =
    R"doc(Returns:
    std::string Provides the URDF model of the attached robot arm as json string

Raises:
    CommandException: if the Control reports an error.
    NetworkException: if the connection is lost, e.g. after a timeout.

)doc";

static const char *mkd_doc_franka_Robot_getUpperJointVelocityLimits =
    R"doc(Computes upper limits for joint velocities based on current joint positions.

Args:
    joint_positions: Joint positions.

Returns:
    Upper limits for joint velocities.

)doc";

static const char *mkd_doc_franka_Robot_impl = R"doc()doc";

static const char *mkd_doc_franka_Robot_isMobileRobot =
    R"doc(Returns whether the connected robot is a mobile robot (TMR).

This is detected at connection time from the robot's URDF.

Returns:
    true if the robot is a mobile robot, false if it is an arm robot.

)doc";

static const char *mkd_doc_franka_Robot_loadModel =
    R"doc(Loads the model library from the robot.

Returns:
    Model instance.

Raises:
    ModelException: if the model library cannot be loaded.
    InvalidOperationException: if called on a mobile robot. Use franka::MobileModel with the URDF
                               from getRobotModel() instead.
    NetworkException: if the connection is lost, e.g. after a timeout.

)doc";

static const char *mkd_doc_franka_Robot_loadModel_2 = R"doc()doc";

static const char *mkd_doc_franka_Robot_operator_assign =
    R"doc(Move-assigns this Robot from another Robot instance.

Args:
    other: Other Robot instance.

Returns:
    Robot instance.

)doc";

static const char *mkd_doc_franka_Robot_operator_assign_2 = R"doc()doc";

static const char *mkd_doc_franka_Robot_read =
    R"doc(Starts a loop for reading the current robot state.

Cannot be executed while a control or motion generator loop is running.

This minimal example will print the robot state 100 times:
```
{.cpp}
franka::Robot robot("robot.franka.de");
size_t count = 0;
robot.read([&count](const franka::RobotState& robot_state) {
  std::cout << robot_state << std::endl;
  return count++ < 100;
});
```



Args:
    read_callback: Callback function for robot state reading.

Raises:
    InvalidOperationException: if a conflicting operation is already running.
    NetworkException: if the connection is lost, e.g. after a timeout.

)doc";

static const char *mkd_doc_franka_Robot_readOnce =
    R"doc(Waits for a robot state update and returns it.

Cannot be executed while a control or motion generator loop is running.

Returns:
    Current robot state.

Raises:
    InvalidOperationException: if a conflicting operation is already running.
    NetworkException: if the connection is lost, e.g. after a timeout.

See also:
    Robot::read for a way to repeatedly receive the robot state.

)doc";

static const char *mkd_doc_franka_Robot_serverVersion =
    R"doc(Returns the software version reported by the connected server.

Returns:
    Software version of the connected server.

)doc";

static const char *mkd_doc_franka_Robot_setCartesianImpedance =
    R"doc(Sets the Cartesian stiffness/compliance (for x, y, z, roll, pitch, yaw) in the internal controller.

The values set using Robot::setCartesianImpedance are used in the direction of the stiffness frame,
which can be set with Robot::setK.

Inputs received by the torque controller are not affected by this setting.

Args:
    K_x: Cartesian impedance values :math:`K_x=(K_{x_{x,y,z}} \in [10,3000] \frac{N}{m},
              K_{x_{R,P,Y}} \in [1,300] \frac{Nm}{rad})`

Raises:
    CommandException: if the Control reports an error.
    NetworkException: if the connection is lost, e.g. after a timeout.

)doc";

static const char *mkd_doc_franka_Robot_setCollisionBehavior =
    R"doc(Changes the collision behavior.

Set separate torque and force boundaries for acceleration/deceleration and constant velocity
movement phases.

Forces or torques between lower and upper threshold are shown as contacts in the RobotState. Forces
or torques above the upper threshold are registered as collision and cause the robot to stop moving.

Args:
    lower_torque_thresholds_acceleration: Contact torque thresholds during
                                               acceleration/deceleration for each joint in
                                               :math:`[Nm]`.
    upper_torque_thresholds_acceleration: Collision torque thresholds during
                                               acceleration/deceleration for each joint in
                                               :math:`[Nm]`.
    lower_torque_thresholds_nominal: Contact torque thresholds for each joint in :math:`[Nm]`.
    upper_torque_thresholds_nominal: Collision torque thresholds for each joint in
                                          :math:`[Nm]`.
    lower_force_thresholds_acceleration: Contact force thresholds during
                                              acceleration/deceleration for :math:`(x,y,z,R,P,Y)` in
                                              :math:`[N]`.
    upper_force_thresholds_acceleration: Collision force thresholds during
                                              acceleration/deceleration for :math:`(x,y,z,R,P,Y)` in
                                              :math:`[N]`.
    lower_force_thresholds_nominal: Contact force thresholds for :math:`(x,y,z,R,P,Y)` in
                                         :math:`[N]`.
    upper_force_thresholds_nominal: Collision force thresholds for :math:`(x,y,z,R,P,Y)` in
                                         :math:`[N]`.

Raises:
    CommandException: if the Control reports an error.
    NetworkException: if the connection is lost, e.g. after a timeout.

See also:
    RobotState::cartesian_contact

See also:
    RobotState::cartesian_collision

See also:
    RobotState::joint_contact

See also:
    RobotState::joint_collision

See also:
    Robot::automaticErrorRecovery for performing a reset after a collision.

)doc";

static const char *mkd_doc_franka_Robot_setCollisionBehavior_2 =
    R"doc(Changes the collision behavior.

Set common torque and force boundaries for acceleration/deceleration and constant velocity movement
phases.

Forces or torques between lower and upper threshold are shown as contacts in the RobotState. Forces
or torques above the upper threshold are registered as collision and cause the robot to stop moving.

Args:
    lower_torque_thresholds: Contact torque thresholds for each joint in :math:`[Nm]`.
    upper_torque_thresholds: Collision torque thresholds for each joint in :math:`[Nm]`.
    lower_force_thresholds: Contact force thresholds for :math:`(x,y,z,R,P,Y)` in :math:`[N]`.
    upper_force_thresholds: Collision force thresholds for :math:`(x,y,z,R,P,Y)` in
                                 :math:`[N]`.

Raises:
    CommandException: if the Control reports an error.
    NetworkException: if the connection is lost, e.g. after a timeout.

See also:
    RobotState::cartesian_contact

See also:
    RobotState::cartesian_collision

See also:
    RobotState::joint_contact

See also:
    RobotState::joint_collision

See also:
    Robot::automaticErrorRecovery for performing a reset after a collision.

)doc";

static const char *mkd_doc_franka_Robot_setEE =
    R"doc(Sets the transformation :math:`^{NE}T_{EE}` from nominal end effector to end effector frame.

The transformation matrix is represented as a vectorized 4x4 matrix in column-major format.

Args:
    NE_T_EE: Vectorized NE-to-EE transformation matrix :math:`^{NE}T_{EE}`, column-major.

Raises:
    CommandException: if the Control reports an error.
    NetworkException: if the connection is lost, e.g. after a timeout.

See also:
    RobotState::NE_T_EE for end effector pose in ne-frame "nominal end effector frame NE".

See also:
    RobotState::O_T_EE for end effector pose in o-frame "world base frame O".

See also:
    RobotState::F_T_EE for end effector pose in f-frame "flange frame F".

)doc";

static const char *mkd_doc_franka_Robot_setGuidingMode =
    R"doc(Locks or unlocks guiding mode movement in (x, y, z, roll, pitch, yaw).

If a flag is set to true, movement is unlocked.

Args:
    guiding_mode: Unlocked movement in (x, y, z, R, P, Y) in guiding mode.
    elbow: True if the elbow is free in guiding mode, false otherwise.

Raises:
    CommandException: if the Control reports an error.
    NetworkException: if the connection is lost, e.g. after a timeout.

Note:
    Guiding mode can be enabled by pressing the two opposing buttons near the robot's flange.

)doc";

static const char *mkd_doc_franka_Robot_setJointImpedance =
    R"doc(Sets the impedance for each joint in the internal controller.

User-provided torques are not affected by this setting.

Args:
    K_theta: Joint impedance values :math:`K_{\theta_{1-7}} = \in [0,14250] \frac{Nm}{rad}`

Raises:
    CommandException: if the Control reports an error.
    NetworkException: if the connection is lost, e.g. after a timeout.

)doc";

static const char *mkd_doc_franka_Robot_setK =
    R"doc(Sets the transformation :math:`^{EE}T_K` from end effector frame to stiffness frame.

The transformation matrix is represented as a vectorized 4x4 matrix in column-major format.

Args:
    EE_T_K: Vectorized EE-to-K transformation matrix :math:`^{EE}T_K`, column-major.

Raises:
    CommandException: if the Control reports an error.
    NetworkException: if the connection is lost, e.g. after a timeout.

See also:
    Robot for an explanation of the stiffness frame.

)doc";

static const char *mkd_doc_franka_Robot_setLoad =
    R"doc(Sets dynamic parameters of a payload.

Args:
    load_mass: Mass of the load in :math:`[kg]`.
    F_x_Cload: Translation from flange to center of mass of load :math:`^Fx_{C_\text{load}}` in
                    :math:`[m]`.
    load_inertia: Inertia matrix :math:`I_\text{load}` in :math:`[kg \times m^2]`, column-
                       major.

Raises:
    CommandException: if the Control reports an error.
    NetworkException: if the connection is lost, e.g. after a timeout.

Note:
    This is not for setting end effector parameters, which have to be set in the administrator's
    interface.

)doc";

static const char *mkd_doc_franka_Robot_startAsyncControl =
    R"doc(Starts a new async motion generator and controller

Args:
    controller_type: defines the type of motion / control that shall be started
    maximum_velocities: maximum joint velocities for the motion generator

Template Args:
    MotionGeneratorType: the franka control type

Returns:
    unique_ptr of ActiveMotionGenerator for the started motion

Raises:
    ControlException: if an error related to torque control or motion generation occurred.
    InvalidOperationException: if a conflicting operation is already running.*
    NetworkException: if the connection is lost,e.g.after a timeout.
    std::invalid_argument: if joint - level torque commands are NaN or infinity.

)doc";

static const char *mkd_doc_franka_Robot_startAsyncJointPositionControl =
    R"doc(Starts a new async joint position motion generator

Args:
    control_type: research_interface::robot::Move::ControllerMode control type for the operation
    maximum_velocities: maximum joint velocities for the motion generator

Returns:
    unique_ptr of ActiveMotionGenerator for the started motion

Raises:
    ControlException: if an error related to torque control or motion generation occurred.
    InvalidOperationException: if a conflicting operation is already running.
    NetworkException: if the connection is lost, e.g. after a timeout.
    std::invalid_argument: if joint-level torque commands are NaN or infinity.

)doc";

static const char *mkd_doc_franka_Robot_startCartesianPoseControl =
    R"doc(Starts a new cartesian position motion generator

Args:
    control_type: research_interface::robot::Move::ControllerMode control type for the operation

Returns:
    unique_ptr of ActiveMotionGenerator for the started motion

Raises:
    ControlException: if an error related to torque control or motion generation occurred.
    InvalidOperationException: if a conflicting operation is already running.
    NetworkException: if the connection is lost, e.g. after a timeout.
    std::invalid_argument: if joint-level torque commands are NaN or infinity.

)doc";

static const char *mkd_doc_franka_Robot_startCartesianVelocityControl =
    R"doc(Starts a new cartesian velocity motion generator

Args:
    control_type: research_interface::robot::Move::ControllerMode control type for the operation

Returns:
    unique_ptr of ActiveMotionGenerator for the started motion

Raises:
    ControlException: if an error related to torque control or motion generation occurred.
    InvalidOperationException: if a conflicting operation is already running.
    NetworkException: if the connection is lost, e.g. after a timeout.
    std::invalid_argument: if joint-level torque commands are NaN or infinity.

)doc";

static const char *mkd_doc_franka_Robot_startControl =
    R"doc(Starts a new motion generator and controller

Args:
    control_mode: defines the type of motion / control that shall be started
    controller_mode: the controller-mode that shall be used

Template Args:
    MotionGeneratorType: the franka control type

Returns:
    unique_ptr of ActiveMotionGenerator for the started motion

Raises:
    ControlException: if an error related to torque control or motion generation occurred.
    InvalidOperationException: if a conflicting operation is already running.*
    NetworkException: if the connection is lost,e.g.after a timeout.
    std::invalid_argument: if joint - level torque commands are NaN or infinity.

)doc";

static const char *mkd_doc_franka_Robot_startJointPositionControl =
    R"doc(Starts a new joint position motion generator

Args:
    control_type: research_interface::robot::Move::ControllerMode control type for the operation

Returns:
    unique_ptr of ActiveMotionGenerator for the started motion

Raises:
    ControlException: if an error related to torque control or motion generation occurred.
    InvalidOperationException: if a conflicting operation is already running.
    NetworkException: if the connection is lost, e.g. after a timeout.
    std::invalid_argument: if joint-level torque commands are NaN or infinity.

)doc";

static const char *mkd_doc_franka_Robot_startJointVelocityControl =
    R"doc(Starts a new joint velocity motion generator

Args:
    control_type: research_interface::robot::Move::ControllerMode control type for the operation

Returns:
    unique_ptr of ActiveMotionGenerator for the started motion

Raises:
    ControlException: if an error related to torque control or motion generation occurred.
    InvalidOperationException: if a conflicting operation is already running.
    NetworkException: if the connection is lost, e.g. after a timeout.
    std::invalid_argument: if joint-level torque commands are NaN or infinity.

)doc";

static const char *mkd_doc_franka_Robot_startTorqueControl =
    R"doc(Starts a new torque controller

Returns:
    unique_ptr of ActiveTorqueControl for the started motion

Raises:
    ControlException: if an error related to torque control or motion generation occurred.
    InvalidOperationException: if a conflicting operation is already running.
    NetworkException: if the connection is lost, e.g. after a timeout.
    std::invalid_argument: if joint-level torque commands are NaN or infinity.

)doc";

static const char *mkd_doc_franka_Robot_stop =
    R"doc(Stops all currently running motions.

If a control or motion generator loop is running in another thread, it will be preempted with a
franka::ControlException.

Raises:
    CommandException: if the Control reports an error.
    NetworkException: if the connection is lost, e.g. after a timeout.

)doc";

static const char *mkd_doc_franka_Torques = R"doc(Stores joint-level torque commands without gravity and friction.)doc";

static const char *mkd_doc_franka_Torques_Torques =
    R"doc(Creates a new Torques instance.

Args:
    torques: Desired joint-level torques without gravity and friction in :math:`[Nm]`.

)doc";

static const char *mkd_doc_franka_Torques_Torques_2 =
    R"doc(Creates a new Torques instance.

Args:
    torques: Desired joint-level torques without gravity and friction in :math:`[Nm]`.

Raises:
    std::invalid_argument: if the given initializer list has an invalid number of arguments.

)doc";

static const char *mkd_doc_franka_Torques_tau_J = R"doc(Desired torques in [Nm].)doc";

static const char *mkd_doc_franka_VacuumGripper =
    R"doc(Maintains a network connection to the vacuum gripper, provides the current vacuum gripper state, and
allows the execution of commands.

Note:
    The members of this class are threadsafe.)doc";

static const char *mkd_doc_franka_VacuumGripperDeviceStatus = R"doc(Vacuum gripper device status.)doc";

static const char *mkd_doc_franka_VacuumGripperDeviceStatus_kGreen = R"doc(Device is working optimally)doc";

static const char *mkd_doc_franka_VacuumGripperDeviceStatus_kOrange =
    R"doc(Device is working but there are severe warnings)doc";

static const char *mkd_doc_franka_VacuumGripperDeviceStatus_kRed = R"doc(Device is not working properly)doc";

static const char *mkd_doc_franka_VacuumGripperDeviceStatus_kYellow =
    R"doc(Device is working but there are warnings)doc";

static const char *mkd_doc_franka_VacuumGripperState =
    R"doc(Describes the vacuum gripper state. For more information check the cobot-pump manual.)doc";

static const char *mkd_doc_franka_VacuumGripperState_actual_power =
    R"doc(Current vacuum gripper actual power. Unit: :math:`[%]`.)doc";

static const char *mkd_doc_franka_VacuumGripperState_device_status = R"doc(Current vacuum gripper device status.)doc";

static const char *mkd_doc_franka_VacuumGripperState_in_control_range =
    R"doc(Vacuum value within in setpoint area.)doc";

static const char *mkd_doc_franka_VacuumGripperState_part_detached =
    R"doc(The part has been detached after a suction cycle)doc";

static const char *mkd_doc_franka_VacuumGripperState_part_present =
    R"doc(Vacuum is over H2 and not yet under H2-h2. For more information check the cobot-pump manual.)doc";

static const char *mkd_doc_franka_VacuumGripperState_time =
    R"doc(Strictly monotonically increasing timestamp since robot start.)doc";

static const char *mkd_doc_franka_VacuumGripperState_vacuum = R"doc(Current system vacuum. Unit: :math:`[mbar]`.)doc";

static const char *mkd_doc_franka_VacuumGripper_ProductionSetupProfile = R"doc(Vacuum production setup profile.)doc";

static const char *mkd_doc_franka_VacuumGripper_ProductionSetupProfile_kP0 = R"doc()doc";

static const char *mkd_doc_franka_VacuumGripper_ProductionSetupProfile_kP1 = R"doc()doc";

static const char *mkd_doc_franka_VacuumGripper_ProductionSetupProfile_kP2 = R"doc()doc";

static const char *mkd_doc_franka_VacuumGripper_ProductionSetupProfile_kP3 = R"doc()doc";

static const char *mkd_doc_franka_VacuumGripper_VacuumGripper =
    R"doc(Establishes a connection with a vacuum gripper connected to a robot.

Args:
    franka_address: IP/hostname of the robot the vacuum gripper is connected to.

Raises:
    NetworkException: if the connection is unsuccessful.
    IncompatibleVersionException: if this version of `libfranka` is not supported.

)doc";

static const char *mkd_doc_franka_VacuumGripper_VacuumGripper_2 =
    R"doc(Move-constructs a new VacuumGripper instance.

Args:
    vacuum_gripper: Other VacuumGripper instance.

)doc";

static const char *mkd_doc_franka_VacuumGripper_VacuumGripper_3 = R"doc(@cond DO_NOT_DOCUMENT)doc";

static const char *mkd_doc_franka_VacuumGripper_dropOff =
    R"doc(Drops the grasped object off.

Args:
    timeout: Dropoff timeout. Unit: :math:`[ms]`.

Returns:
    True if command was successful, false otherwise.

Raises:
    CommandException: if an error occurred.
    NetworkException: if the connection is lost, e.g. after a timeout.

)doc";

static const char *mkd_doc_franka_VacuumGripper_network = R"doc(@endcond)doc";

static const char *mkd_doc_franka_VacuumGripper_operator_assign =
    R"doc(Move-assigns this VacuumGripper from another VacuumGripper instance.

Args:
    vacuum_gripper: Other VacuumGripper instance.

Returns:
    Model instance.

)doc";

static const char *mkd_doc_franka_VacuumGripper_operator_assign_2 = R"doc()doc";

static const char *mkd_doc_franka_VacuumGripper_readOnce =
    R"doc(Waits for a vacuum gripper state update and returns it.

Returns:
    Current vacuum gripper state.

Raises:
    NetworkException: if the connection is lost, e.g. after a timeout.
    InvalidOperationException: if another readOnce is already running.

)doc";

static const char *mkd_doc_franka_VacuumGripper_ri_version = R"doc()doc";

static const char *mkd_doc_franka_VacuumGripper_serverVersion =
    R"doc(Returns the software version reported by the connected server.

Returns:
    Software version of the connected server.

)doc";

static const char *mkd_doc_franka_VacuumGripper_stop =
    R"doc(Stops a currently running vacuum gripper vacuum or drop off operation.

Returns:
    True if command was successful, false otherwise.

Raises:
    CommandException: if an error occurred.
    NetworkException: if the connection is lost, e.g. after a timeout.

)doc";

static const char *mkd_doc_franka_VacuumGripper_vacuum =
    R"doc(Vacuums an object.

Args:
    vacuum: Setpoint for control mode. Unit: :math:`[10*mbar]`.
    timeout: Vacuum timeout. Unit: :math:`[ms]`.
    profile: Production setup profile P0 to P3. Default: P0.

Returns:
    True if the vacuum has been established, false otherwise.

Raises:
    CommandException: if an error occurred.
    NetworkException: if the connection is lost, e.g. after a timeout.

)doc";

static const char *mkd_doc_franka_cartesianLowpassFilter =
    R"doc(Applies a first-order low-pass filter to the translation and spherical linear interpolation to the
rotation of a transformation matrix which represents a Cartesian Motion.

Args:
    sample_time: Sample time constant
    current_signal_value: Current Cartesian transformation matrix to be filtered
    last_signal_value: Cartesian transformation matrix from the previous time step
    cutoff_frequency: Cutoff frequency of the low-pass filter

Returns:
    Filtered Cartesian transformation matrix.

Raises:
    std::invalid_argument: if elements of current_signal_value is infinite or NaN.
                           std::invalid_argument: if elements of last_signal_value is infinite or
                           NaN. std::invalid_argument: if cutoff_frequency is zero, negative,
                           infinite or NaN. std::invalid_argument: if sample_time is negative,
                           infinite or NaN.

)doc";

static const char *mkd_doc_franka_checkElbow =
    R"doc(Checks if all elements of the elbow vector are finite and if the elbow configuration is valid

Args:
    elbow: the elbow vector to check

)doc";

static const char *mkd_doc_franka_checkFinite =
    R"doc(Checks if all elements of an array of the size N have a finite value

Args:
    array: the array to be checked

Template Args:
    N: the size of the array

)doc";

static const char *mkd_doc_franka_checkMatrix =
    R"doc(Checks if all elements of the transformation matrix are finite and if it is a homogeneous
transformation

Args:
    transform: the transformation matrix to check

)doc";

static const char *mkd_doc_franka_computeLowerLimitsJointVelocity =
    R"doc(Computes the minimum joint velocity based on joint position

Args:
    q: joint position.

Returns:
    Lower limits of joint velocity at the given joint position.

Note:
    The implementation is based on
    https://frankarobotics.github.io/docs/control_parameters.html#limits-for-franka-research-3.

Deprecated:
    This function is deprecated and will be removed in future versions. Use
    `Robot::getlowerJointVelocityLimits(const std::array<double, 7UL> &joint_positions)` instead (if
    applicable).

)doc";

static const char *mkd_doc_franka_computeUpperLimitsJointVelocity =
    R"doc(Computes the maximum joint velocity based on joint position

Args:
    q: joint position.

Returns:
    Upper limits of joint velocity at the given joint position.

Note:
    The implementation is based on
    https://frankarobotics.github.io/docs/control_parameters.html#limits-for-franka-research-3.

Deprecated:
    This function is deprecated and will be removed in future versions. Use
    `Robot::getUpperJointVelocityLimits(const std::array<double, 7UL> &joint_positions)` instead (if
    applicable).

)doc";

static const char *mkd_doc_franka_hasRealtimeKernel =
    R"doc(Determines whether the current OS kernel is a realtime kernel.

On Linux, this checks for the existence of `/sys/kernel/realtime`. On Windows, this always returns
true.

Returns:
    True if running a realtime kernel, false otherwise.

)doc";

static const char *mkd_doc_franka_isHomogeneousTransformation =
    R"doc(Determines whether the given array represents a valid homogeneous transformation matrix.

Args:
    transform: 4x4 matrix in column-major format.

Returns:
    True if the array represents a homogeneous transformation matrix, otherwise false.

)doc";

static const char *mkd_doc_franka_isValidElbow =
    R"doc(Determines whether the given elbow configuration is valid or not.

Args:
    elbow: Elbow configuration.

Returns:
    True if valid, otherwise false.

)doc";

static const char *mkd_doc_franka_limitRate =
    R"doc(Limits the rate of an input vector of per-joint commands considering the maximum allowed time
derivatives.

Args:
    max_derivatives: Per-joint maximum allowed time derivative.
    commanded_values: Commanded values of the current time step.
    last_commanded_values: Commanded values of the previous time step.

Returns:
    Rate-limited vector of desired values.

Raises:
    std::invalid_argument: if commanded_values are infinite or NaN.

Note:
    FCI filters must be deactivated to work properly.

)doc";

static const char *mkd_doc_franka_limitRate_2 =
    R"doc(Limits the rate of a desired joint velocity considering the limits provided.

Args:
    upper_limits_velocity: Upper limits of allowed velocity.
    lower_limits_velocity: Lower limits of allowed velocity.
    max_acceleration: Maximum allowed acceleration.
    max_jerk: Maximum allowed jerk.
    commanded_velocity: Commanded joint velocity of the current time step.
    last_commanded_velocity: Commanded joint velocity of the previous time step.
    last_commanded_acceleration: Commanded joint acceleration of the previous time step.

Returns:
    Rate-limited desired joint velocity.

Raises:
    std::invalid_argument: if commanded_velocity is infinite or NaN.

Note:
    FCI filters must be deactivated to work properly.

)doc";

static const char *mkd_doc_franka_limitRate_3 =
    R"doc(Limits the rate of a desired joint position considering the limits provided.

Args:
    upper_limits_velocity: Upper limits of allowed velocity.
    lower_limits_velocity: Lower limits of allowed velocity.
    max_acceleration: Maximum allowed acceleration.
    max_jerk: Maximum allowed jerk.
    commanded_position: Commanded joint position of the current time step.
    last_commanded_position: Commanded joint position of the previous time step.
    last_commanded_velocity: Commanded joint velocity of the previous time step.
    last_commanded_acceleration: Commanded joint acceleration of the previous time step.

Returns:
    Rate-limited desired joint position.

Raises:
    std::invalid_argument: if commanded_position is infinite or NaN.

Note:
    FCI filters must be deactivated to work properly.

)doc";

static const char *mkd_doc_franka_limitRate_4 =
    R"doc(Limits the rate of a desired joint velocity considering the limits provided.

Args:
    upper_limits_velocity: Per-joint upper limits of allowed velocity.
    lower_limits_velocity: Per-joint lower limits of allowed velocity.
    max_acceleration: Per-joint maximum allowed acceleration.
    max_jerk: Per-joint maximum allowed jerk.
    commanded_velocities: Commanded joint velocity of the current time step.
    last_commanded_velocities: Commanded joint velocities of the previous time step.
    last_commanded_accelerations: Commanded joint accelerations of the previous time step.

Returns:
    Rate-limited vector of desired joint velocities.

Raises:
    std::invalid_argument: if commanded_velocities are infinite or NaN.

Note:
    FCI filters must be deactivated to work properly.

)doc";

static const char *mkd_doc_franka_limitRate_5 =
    R"doc(Limits the rate of a desired joint position considering the limits provided.

Args:
    upper_limits_velocity: Per-joint upper limits of allowed velocity.
    lower_limits_velocity: Per-joint lower limits of allowed velocity.
    max_acceleration: Per-joint maximum allowed acceleration.
    max_jerk: Per-joint maximum allowed jerk.
    commanded_positions: Commanded joint positions of the current time step.
    last_commanded_positions: Commanded joint positions of the previous time step.
    last_commanded_velocities: Commanded joint velocities of the previous time step.
    last_commanded_accelerations: Commanded joint accelerations of the previous time step.

Returns:
    Rate-limited vector of desired joint positions.

Raises:
    std::invalid_argument: if commanded_positions are infinite or NaN.

Note:
    FCI filters must be deactivated to work properly.

)doc";

static const char *mkd_doc_franka_limitRate_6 =
    R"doc(Limits the rate of a desired Cartesian velocity considering the limits provided.

Args:
    max_translational_velocity: Maximum translational velocity.
    max_translational_acceleration: Maximum translational acceleration.
    max_translational_jerk: Maximum translational jerk.
    max_rotational_velocity: Maximum rotational velocity.
    max_rotational_acceleration: Maximum rotational acceleration.
    max_rotational_jerk: Maximum rotational jerk.
    O_dP_EE_c: Commanded end effector twist of the current time step.
    last_O_dP_EE_c: Commanded end effector twist of the previous time step.
    last_O_ddP_EE_c: Commanded end effector acceleration of the previous time step.

Returns:
    Rate-limited desired end effector twist.

Raises:
    std::invalid_argument: if an element of O_dP_EE_c is infinite or NaN.

Note:
    FCI filters must be deactivated to work properly.

)doc";

static const char *mkd_doc_franka_limitRate_7 =
    R"doc(Limits the rate of a desired Cartesian pose considering the limits provided.

Args:
    max_translational_velocity: Maximum translational velocity.
    max_translational_acceleration: Maximum translational acceleration.
    max_translational_jerk: Maximum translational jerk.
    max_rotational_velocity: Maximum rotational velocity.
    max_rotational_acceleration: Maximum rotational acceleration.
    max_rotational_jerk: Maximum rotational jerk.
    O_T_EE_c: Commanded pose of the current time step.
    last_O_T_EE_c: Commanded pose of the previous time step.
    last_O_dP_EE_c: Commanded end effector twist of the previous time step.
    last_O_ddP_EE_c: Commanded end effector acceleration of the previous time step.

Returns:
    Rate-limited desired pose.

Raises:
    std::invalid_argument: if an element of O_T_EE_c is infinite or NaN.

Note:
    FCI filters must be deactivated to work properly.

)doc";

static const char *mkd_doc_franka_lowpassFilter =
    R"doc(Applies a first-order low-pass filter

Args:
    sample_time: Sample time constant
    current_signal_value: Current value of the signal to be filtered
    last_signal_value: Value of the signal to be filtered in the previous time step
    cutoff_frequency: Cutoff frequency of the low-pass filter

Returns:
    Filtered value.

Raises:
    std::invalid_argument: if current_signal_value is infinite or NaN. std::invalid_argument: if
                           last_signal_value is infinite or NaN. std::invalid_argument: if
                           cutoff_frequency is zero, negative, infinite or NaN.
                           std::invalid_argument: if sample_time is negative, infinite or NaN.

)doc";

static const char *mkd_doc_franka_operator_inc =
    R"doc(Post-increments the given Frame by one.

For example, Frame::kJoint2++ results in Frame::kJoint3.

Args:
    frame: Frame to increment.

Returns:
    Original Frame.

)doc";

static const char *mkd_doc_franka_operator_lshift =
    R"doc(Streams the errors as JSON array.

Args:
    ostream: Ostream instance
    errors: Errors struct instance to stream

Returns:
    Ostream instance

)doc";

static const char *mkd_doc_franka_operator_lshift_2 =
    R"doc(Streams the gripper state as JSON object: {"field_name_1": value, "field_name_2": value, ...}

Args:
    ostream: Ostream instance
    gripper_state: GripperState struct instance to stream

Returns:
    Ostream instance

)doc";

static const char *mkd_doc_franka_operator_lshift_3 =
    R"doc(Streams the robot state as JSON object: {"field_name_1": [0,0,0,0,0,0,0], "field_name_2":
[0,0,0,0,0,0], ...}

Args:
    ostream: Ostream instance
    robot_state: RobotState instance to stream

Returns:
    Ostream instance

)doc";

static const char *mkd_doc_franka_operator_lshift_4 =
    R"doc(Streams RobotMode in human-readable form

Args:
    ostream: Ostream instance
    robot_mode: RobotMode to stream

Returns:
    Ostream instance

)doc";

static const char *mkd_doc_franka_operator_lshift_5 =
    R"doc(Streams the vacuum gripper state as JSON object: {"field_name_1": value, "field_name_2": value, ...}

Args:
    ostream: Ostream instance
    vacuum_gripper_state: VacuumGripperState struct instance to stream

Returns:
    Ostream instance

)doc";

static const char *mkd_doc_franka_operator_mul =
    R"doc(Performs multiplication.

Args:
    lhs: Left-hand side of the multiplication.
    rhs: Right-hand side of the multiplication.

Returns:
    Result of the multiplication.

)doc";

static const char *mkd_doc_franka_setCurrentThreadToHighestSchedulerPriority =
    R"doc(Sets the current thread to the highest possible scheduler priority.

Args:
    error_message: Contains an error message if the scheduler priority cannot be set
                         successfully.

Returns:
    True if successful, false otherwise.

)doc";

static const char *mkd_doc_franky_CartesianImpedanceBase =
    R"doc(Base class for client-side cartesian impedance motions.

This motion implements a cartesian impedance controller on the client side and does not use Franka's
internal impedance controller. Instead, it uses Franka's internal torque controller and calculates
the torques itself.)doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_CartesianImpedanceBase =
    R"doc(Args:
    target: The target pose.
    params: Parameters for the motion.
    gains_time_constant: Smoothing time constant for gain transitions [s]. Default 0.1s.

)doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_Params = R"doc(Parameters for the impedance motion.)doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_Params_damping =
    R"doc(Cartesian damping matrix. If unset, critical damping is used.)doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_Params_force_constraints =
    R"doc(Per-axis force/torque constraints [N, Nm]. nullopt on an axis means unconstrained.)doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_Params_friction =
    R"doc(Per-joint friction feedforward. Defaults to zero (disabled).)doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_Params_nullspace_tasks =
    R"doc(Nullspace objectives.

Each task contributes a joint-space torque that is summed and projected into the Jacobian nullspace.
At most one task of each concrete type is supported because runtime-adjustable gains are shared by
task type.)doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_Params_rotational_error_clip =
    R"doc(Maximum absolute Cartesian orientation error [rad] used by the task-space controller.

The rotational error is clamped elementwise in the base frame before the impedance wrench is
computed. This bounds the commanded Cartesian torque.)doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_Params_safety =
    R"doc(Shared torque safety limits and soft joint-limit repulsion settings.)doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_Params_stiffness =
    R"doc(Cartesian stiffness matrix [N/m, Nm/rad], ordered [x, y, z, rx, ry, rz].)doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_Params_translational_error_clip =
    R"doc(Maximum absolute Cartesian position error [m] used by the task-space controller.

The translational error is clamped elementwise before the impedance wrench is computed. This bounds
the commanded Cartesian force when the reference jumps or contact prevents the end effector from
reaching the target.)doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_Params_validate =
    R"doc(Throw std::invalid_argument if any parameter is out of range.)doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_base_params = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_computeCommand = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_criticalDamping = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_critical_damping =
    R"doc(Cached critical damping = defaultCartesianImpedanceDamping(current_stiffness_).)doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_critical_damping_stiffness =
    R"doc(Stiffness for which critical_damping_ was last computed. Lets criticalDamping() skip the
eigendecomposition while the stiffness is unchanged.)doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_current_damping = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_current_nullspace_gains = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_current_stiffness = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_gains_handle = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_gains_time_constant = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_getGains =
    R"doc(Get a copy of the current target impedance gains.

Mutating the returned object has no effect on the motion; pass it to setGains to apply changes.

)doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_getNullspaceGains =
    R"doc(Get a copy of the current target nullspace gains.

Mutating the returned object has no effect on the motion; pass it to setNullspaceGains to apply
changes.

)doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_nullspace_gains_handle = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_params = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_setGains =
    R"doc(Set the target impedance gains.

The gains are validated and then smoothed in the control loop via exponential interpolation.

Args:
    gains: The new target gains.

)doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_setNullspaceGains =
    R"doc(Set the target nullspace gains.

The gains are smoothed in the control loop via exponential interpolation.

Args:
    gains: The new target nullspace gains.

)doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_target = R"doc(The target pose of the motion.)doc";

static const char *mkd_doc_franky_CartesianImpedanceBase_target_2 = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceGains =
    R"doc(Runtime-adjustable stiffness and damping gains for Cartesian impedance motions.)doc";

static const char *mkd_doc_franky_CartesianImpedanceGains_CartesianImpedanceGains = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceGains_CartesianImpedanceGains_2 = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceGains_damping =
    R"doc(Cartesian damping matrix. If unset, critical damping is used.)doc";

static const char *mkd_doc_franky_CartesianImpedanceGains_diagonal =
    R"doc(Create gains from per-axis diagonal entries.

Args:
    stiffness: Per-axis stiffness [N/m, Nm/rad], ordered [x, y, z, rx, ry, rz].
    damping: Per-axis damping [Ns/m, Nms/rad], ordered [x, y, z, rx, ry, rz]. If unset, critical
             damping is used.

)doc";

static const char *mkd_doc_franky_CartesianImpedanceGains_isotropic =
    R"doc(Create gains with uniform translational and rotational components.

Args:
    translational_stiffness: Stiffness applied to all translational axes [N/m].
    rotational_stiffness: Stiffness applied to all rotational axes [Nm/rad].
    translational_damping: Damping applied to all translational axes [Ns/m]. If unset, critical
                           damping is used.
    rotational_damping: Damping applied to all rotational axes [Nms/rad]. If unset, critical damping
                        is used.

)doc";

static const char *mkd_doc_franky_CartesianImpedanceGains_stiffness =
    R"doc(Cartesian stiffness matrix [N/m, Nm/rad], ordered [x, y, z, rx, ry, rz].)doc";

static const char *mkd_doc_franky_CartesianImpedanceGains_validate =
    R"doc(Throw std::invalid_argument unless gains are symmetric positive semidefinite matrices.)doc";

static const char *mkd_doc_franky_CartesianImpedanceMotion =
    R"doc(Cartesian impedance motion.

This motion implements a cartesian impedance controller on the client side and does not use Franka's
internal impedance controller. Instead, it uses Franka's internal torque controller and calculates
the torques itself.

Analogous to JointImpedanceMotion, this motion regulates toward a fixed target pose (and optional
target twist) and does not terminate on its own. It runs until it is preempted or stopped.)doc";

static const char *mkd_doc_franky_CartesianImpedanceMotion_CartesianImpedanceMotion =
    R"doc(Args:
    target: The target pose.

)doc";

static const char *mkd_doc_franky_CartesianImpedanceMotion_CartesianImpedanceMotion_2 =
    R"doc(Args:
    target: The target pose.
    params: Parameters for the motion.

)doc";

static const char *mkd_doc_franky_CartesianImpedanceMotion_CartesianImpedanceMotion_3 =
    R"doc(Args:
    target: The target pose.
    target_twist: The target twist in the base frame. The damping term acts on twist error rather
                  than resisting all motion toward zero.

)doc";

static const char *mkd_doc_franky_CartesianImpedanceMotion_CartesianImpedanceMotion_4 =
    R"doc(Args:
    target: The target pose.
    target_twist: The target twist in the base frame. The damping term acts on twist error rather
                  than resisting all motion toward zero.
    params: Parameters for the motion.

)doc";

static const char *mkd_doc_franky_CartesianImpedanceMotion_Params =
    R"doc(Parameters for the Cartesian impedance motion.

See also:
    CartesianImpedanceBase::Params)doc";

static const char *mkd_doc_franky_CartesianImpedanceMotion_Params_target_type =
    R"doc(The type of the target reference (relative or absolute).)doc";

static const char *mkd_doc_franky_CartesianImpedanceMotion_initImpl = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceMotion_nextCommandImpl = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceMotion_original_target =
    R"doc(The target pose as passed to the constructor, before any relative-target resolution at motion start.)doc";

static const char *mkd_doc_franky_CartesianImpedanceMotion_params = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceMotion_params_2 = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceMotion_target_twist = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceMotion_target_twist_2 = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceTrackingMotion =
    R"doc(Cartesian impedance tracking motion.

This motion keeps the same Cartesian impedance controller alive while reading the latest reference
from a handle or callback every control cycle.)doc";

static const char *mkd_doc_franky_CartesianImpedanceTrackingMotion_CartesianImpedanceTrackingMotion = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceTrackingMotion_CartesianImpedanceTrackingMotion_2 = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceTrackingMotion_getReference =
    R"doc(Get a copy of the last commanded Cartesian reference, or nullopt if no reference has been set yet.

Mutating the returned object has no effect on the motion; pass it to setReference to apply changes.

)doc";

static const char *mkd_doc_franky_CartesianImpedanceTrackingMotion_initImpl = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceTrackingMotion_nextCommandImpl = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceTrackingMotion_params = R"doc(The parameters of the motion.)doc";

static const char *mkd_doc_franky_CartesianImpedanceTrackingMotion_reference_callback = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceTrackingMotion_reference_handle = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceTrackingMotion_setReference =
    R"doc(Set the Cartesian reference tracked by the controller.

The reference is validated and picked up by the control loop in the next cycle.

Args:
    reference: The new reference.

)doc";

static const char *mkd_doc_franky_CartesianImpedanceTrackingMotion_target_acceleration = R"doc()doc";

static const char *mkd_doc_franky_CartesianImpedanceTrackingMotion_target_twist = R"doc()doc";

static const char *mkd_doc_franky_CartesianMotion = R"doc(Cartesian motion with a single target.)doc";

static const char *mkd_doc_franky_CartesianMotion_CartesianMotion =
    R"doc(Construct a Cartesian motion.

Args:
    target: The target Cartesian state.
    reference_type: The reference type (absolute or relative). An absolute target is defined in the
                    robot's base frame, a relative target is defined in the current end-effector
                    frame.
    relative_dynamics_factor: The relative dynamics factor for this motion. The factor will get
                              multiplied with the robot's global dynamics factor to get the actual
                              dynamics factor for this motion.
    return_when_finished: Whether to end the motion when the target is reached or keep holding the
                          last target.
    ee_frame: The end-effector frame for which the target is defined. This is a transformation from
              the configured end-effector frame to the end-effector frame the target is defined for.

)doc";

static const char *mkd_doc_franky_CartesianReference =
    R"doc(Cartesian impedance reference expressed in the base frame.)doc";

static const char *mkd_doc_franky_CartesianReference_target = R"doc(Desired end-effector pose.)doc";

static const char *mkd_doc_franky_CartesianReference_target_acceleration =
    R"doc(Desired end-effector acceleration in the base frame.

When present, the controller adds a model-based inertial feedforward wrench Lambda(q) *
target_acceleration before mapping through J^T.)doc";

static const char *mkd_doc_franky_CartesianReference_target_twist =
    R"doc(Desired end-effector twist in the base frame.

When present, the damping term acts on twist error rather than resisting all motion toward zero.)doc";

static const char *mkd_doc_franky_CartesianReference_validate =
    R"doc(Throw std::invalid_argument if any value is non-finite.)doc";

static const char *mkd_doc_franky_CartesianState =
    R"doc(Cartesian state of a robot.

This class encapsulates the cartesian state of a robot, which comprises the end effector pose and
the end effector velocity.)doc";

static const char *mkd_doc_franky_CartesianState_CartesianState = R"doc()doc";

static const char *mkd_doc_franky_CartesianState_CartesianState_2 =
    R"doc(Args:
    pose: The pose of the end effector.
    velocity: The velocity of the end effector.

)doc";

static const char *mkd_doc_franky_CartesianState_CartesianState_3 = R"doc()doc";

static const char *mkd_doc_franky_CartesianState_CartesianState_4 = R"doc()doc";

static const char *mkd_doc_franky_CartesianState_changeEndEffectorFrame =
    R"doc(Change the end effector frame of the state by the given affine transform.

Args:
    transform: The pose of the new end-effector in the frame of the old end-effector.

Returns:
    The state with a new end-effector frame.

)doc";

static const char *mkd_doc_franky_CartesianState_pose = R"doc(Pose component of the state.)doc";

static const char *mkd_doc_franky_CartesianState_pose_2 = R"doc()doc";

static const char *mkd_doc_franky_CartesianState_transformWith =
    R"doc(Transform the frame of the state by applying the given affine transform.

Args:
    transform: The transformation to apply.

Returns:
    The state after the transformation.

)doc";

static const char *mkd_doc_franky_CartesianState_velocity = R"doc(Velocity component of the state.)doc";

static const char *mkd_doc_franky_CartesianState_velocity_2 = R"doc()doc";

static const char *mkd_doc_franky_CartesianVelocityMotion =
    R"doc(Cartesian velocity motion with a single target velocity.)doc";

static const char *mkd_doc_franky_CartesianVelocityMotion_CartesianVelocityMotion =
    R"doc(Construct a Cartesian velocity motion.

Args:
    target: The target Cartesian velocity.
    duration: For how long the target is held after it has been reached. Default is 1s.
    relative_dynamics_factor: The relative dynamics factor for this motion. The factor will get
                              multiplied with the robot's global dynamics factor to get the actual
                              dynamics factor for this motion.
    ee_frame: The end-effector frame for which the target is defined. This is a transformation from
              the configured end-effector frame to the end-effector frame the target is defined for.

)doc";

static const char *mkd_doc_franky_CartesianVelocityWaypointMotion =
    R"doc(Cartesian velocity waypoint motion.

This motion follows multiple consecutive cartesian velocity targets in a time-optimal way.)doc";

static const char *mkd_doc_franky_CartesianVelocityWaypointMotion_CartesianVelocityWaypointMotion =
    R"doc(Args:
    waypoints: Waypoints to follow.
    relative_dynamics_factor: The relative dynamics factor for this motion. This factor will get
                              multiplied with the robot's global dynamics factor to get the actual
                              dynamics factor for this motion.
    ee_frame: The end-effector frame for which the target is defined. This is a transformation from
              the configured end-effector frame to the end-effector frame the target is defined for.

)doc";

static const char *mkd_doc_franky_CartesianVelocityWaypointMotion_checkWaypoint = R"doc()doc";

static const char *mkd_doc_franky_CartesianVelocityWaypointMotion_ee_frame =
    R"doc(The end-effector frame for which the targets are defined.)doc";

static const char *mkd_doc_franky_CartesianVelocityWaypointMotion_ee_frame_2 = R"doc()doc";

static const char *mkd_doc_franky_CartesianVelocityWaypointMotion_getAbsoluteInputLimits = R"doc()doc";

static const char *mkd_doc_franky_CartesianVelocityWaypointMotion_getControlSignal = R"doc()doc";

static const char *mkd_doc_franky_CartesianVelocityWaypointMotion_getDesiredState = R"doc()doc";

static const char *mkd_doc_franky_CartesianVelocityWaypointMotion_initWaypointMotion = R"doc()doc";

static const char *mkd_doc_franky_CartesianVelocityWaypointMotion_last_elbow_pos = R"doc()doc";

static const char *mkd_doc_franky_CartesianVelocityWaypointMotion_last_elbow_vel = R"doc()doc";

static const char *mkd_doc_franky_CartesianVelocityWaypointMotion_setNewWaypoint = R"doc()doc";

static const char *mkd_doc_franky_CartesianVelocityWaypointMotion_vec_cart_rot_elbow = R"doc()doc";

static const char *mkd_doc_franky_CartesianWaypointMotion =
    R"doc(Cartesian waypoint motion.

This motion follows multiple consecutive cartesian targets in a time-optimal way.)doc";

static const char *mkd_doc_franky_CartesianWaypointMotion_CartesianWaypointMotion =
    R"doc(Args:
    waypoints: Waypoints to follow.
    relative_dynamics_factor: The relative dynamics factor for this motion. This factor will get
                              multiplied with the robot's global dynamics factor to get the actual
                              dynamics factor for this motion.
    return_when_finished: Whether to end the motion when the last waypoint is reached or keep
                          holding the last target.
    ee_frame: The end-effector frame for which the target is defined. This is a transformation from
              the configured end-effector frame to the end-effector frame the target is defined for.

)doc";

static const char *mkd_doc_franky_CartesianWaypointMotion_ee_frame =
    R"doc(The end-effector frame for which the targets are defined.)doc";

static const char *mkd_doc_franky_CartesianWaypointMotion_ee_frame_2 = R"doc()doc";

static const char *mkd_doc_franky_CartesianWaypointMotion_getAbsoluteInputLimits = R"doc()doc";

static const char *mkd_doc_franky_CartesianWaypointMotion_getControlSignal = R"doc()doc";

static const char *mkd_doc_franky_CartesianWaypointMotion_getDesiredState = R"doc()doc";

static const char *mkd_doc_franky_CartesianWaypointMotion_initWaypointMotion = R"doc()doc";

static const char *mkd_doc_franky_CartesianWaypointMotion_ref_frame = R"doc()doc";

static const char *mkd_doc_franky_CartesianWaypointMotion_setNewWaypoint = R"doc()doc";

static const char *mkd_doc_franky_CartesianWaypointMotion_target_state = R"doc()doc";

static const char *mkd_doc_franky_CartesianWaypointMotion_vec_cart_rot_elbow = R"doc()doc";

static const char *mkd_doc_franky_Condition =
    R"doc(A condition on the robot state.

This class defines a condition on the robot state, which can be used to define a condition for a
reaction in a motion. Conditions support logical operations (conjunction, disjunction, equality, and
negation) and can be combined to form more complex conditions.)doc";

static const char *mkd_doc_franky_Condition_Condition =
    R"doc(Args:
    check_func: A function that returns true if the condition is met.
    repr: A string representation of the condition.

)doc";

static const char *mkd_doc_franky_Condition_Condition_2 =
    R"doc(Implicit constructor for constant conditions.

Args:
    constant_value: The constant value of the condition.

)doc";

static const char *mkd_doc_franky_Condition_check_func = R"doc()doc";

static const char *mkd_doc_franky_Condition_operator_call =
    R"doc(Check if the condition is met.

Args:
    robot_state: The current robot state.
    rel_time: The time since the start of the current motion.
    abs_time: The time since the start of the current chain of motions. This value measures the time
              since the robot started moving, and is only reset if a motion expires without being
              replaced by a new motion.

Returns:
    True if the condition is met.

)doc";

static const char *mkd_doc_franky_Condition_repr = R"doc(The string representation of the condition.)doc";

static const char *mkd_doc_franky_Condition_repr_2 = R"doc()doc";

static const char *mkd_doc_franky_ControlSignalType = R"doc(Type of control signal.)doc";

static const char *mkd_doc_franky_ControlSignalType_CartesianPose = R"doc()doc";

static const char *mkd_doc_franky_ControlSignalType_CartesianVelocities = R"doc()doc";

static const char *mkd_doc_franky_ControlSignalType_JointPositions = R"doc()doc";

static const char *mkd_doc_franky_ControlSignalType_JointVelocities = R"doc()doc";

static const char *mkd_doc_franky_ControlSignalType_Torques = R"doc()doc";

static const char *mkd_doc_franky_DynamicsLimit =
    R"doc(A template class representing a dynamics limit with a maximum value.

This class provides functionality to store and manage a dynamic limit, including setting and getting
the value of the limit while ensuring that it cannot be written while the robot is in control.

Template Args:
    LimitType: The type of the limit value (e.g., double, int).)doc";

static const char *mkd_doc_franky_DynamicsLimit_DynamicsLimit =
    R"doc(Constructor for DynamicsLimit.

Initializes a new instance of the DynamicsLimit class with the provided values. The maximum value is
used as default value by this constructor.

Args:
    desc: A string description of the limit (e.g., "joint velocity").
    max_val: The maximum allowable value for this limit.
    write_mutex: A shared pointer to the mutex used for synchronizing writes to the limit.
    can_write_condition: A function that returns true if the limit can be written to.

)doc";

static const char *mkd_doc_franky_DynamicsLimit_DynamicsLimit_2 =
    R"doc(Constructor for DynamicsLimit.

Initializes a new instance of the DynamicsLimit class with the provided values.

Args:
    desc: A string description of the limit (e.g., "joint velocity").
    max_val: The maximum allowable value for this limit.
    write_mutex: A shared pointer to the mutex used for synchronizing writes to the limit.
    can_write_condition: A function that returns true if the limit can be written to.
    default_val: The default value for this limit.

)doc";

static const char *mkd_doc_franky_DynamicsLimit_can_write_condition =
    R"doc(Function to check if writing is allowed.)doc";

static const char *mkd_doc_franky_DynamicsLimit_check =
    R"doc(Validate the new value before setting it.

This function checks whether the given value is within acceptable bounds for the limit. It is called
before setting the limit to ensure correctness.

Args:
    value: The value to check.

)doc";

static const char *mkd_doc_franky_DynamicsLimit_desc =
    R"doc(Description of this limit.

This string provides a human-readable description of the limit, such as "joint velocity" or "maximum
load".)doc";

static const char *mkd_doc_franky_DynamicsLimit_get =
    R"doc(Get the current value of the limit.

Retrieves the current value stored in this limit.

Returns:
    The current value of the limit.

)doc";

static const char *mkd_doc_franky_DynamicsLimit_max =
    R"doc(The maximum value this limit can take as defined by Franka.

This value represents the maximum boundary for the limit, beyond which the value cannot be set.)doc";

static const char *mkd_doc_franky_DynamicsLimit_set =
    R"doc(Set a new value for the limit.

Sets the value of the limit, but only if the robot is not in control. Throws a runtime_error if the
condition for writing to the limit is not met.

Args:
    value: The new value to set for the limit.

Raises:
    std::runtime_error: if the limit cannot be set due to the robot being in control.

)doc";

static const char *mkd_doc_franky_DynamicsLimit_setFrom =
    R"doc(Set a new value for the limit with a different type.

This version of `set` allows setting the value with a different type, with appropriate type
conversions.

Args:
    value: The new value to set for the limit.

Template Args:
    AlternativeType: The type of the alternative value to set.

)doc";

static const char *mkd_doc_franky_DynamicsLimit_value = R"doc(Current value of the limit.)doc";

static const char *mkd_doc_franky_DynamicsLimit_write_mutex = R"doc(Mutex for synchronizing writes to the limit.)doc";

static const char *mkd_doc_franky_ElbowState =
    R"doc(Elbow state of the robot.

This class encapsulates the elbow state of a robot, which comprises the position of the 3rd joint
and the flip direction of the 4th joint. For details see https://frankarobotics.github.io/libfranka/
0.15.3/structfranka_1_1RobotState.html#a43485841c427d70e7f36a912cc3116d1)doc";

static const char *mkd_doc_franky_ElbowState_ElbowState =
    R"doc(Construct an elbow state with the given joint position and optional flip direction.

Args:
    joint_3_pos: The position of the 3rd joint.
    joint_4_flip: The flip direction of the 4th joint.

)doc";

static const char *mkd_doc_franky_ElbowState_ElbowState_2 =
    R"doc(Construct an elbow state from an array containing joint position and flip direction.

Args:
    elbow_state: The joint position and flip direction as a double array.

)doc";

static const char *mkd_doc_franky_ElbowState_ElbowState_3 = R"doc()doc";

static const char *mkd_doc_franky_ElbowState_ElbowState_4 = R"doc()doc";

static const char *mkd_doc_franky_ElbowState_joint_3_pos = R"doc(The position of the 3rd joint.)doc";

static const char *mkd_doc_franky_ElbowState_joint_3_pos_2 = R"doc()doc";

static const char *mkd_doc_franky_ElbowState_joint_4_flip = R"doc(The flip direction of the 4th joint.)doc";

static const char *mkd_doc_franky_ElbowState_joint_4_flip_2 = R"doc()doc";

static const char *mkd_doc_franky_ElbowState_to_array =
    R"doc(Get the joint position and flip direction as an array.

Args:
    default_flip_direction: The default flip direction to use if not explicitly set.

Returns:
    std::array containing the joint position and flip direction.

)doc";

static const char *mkd_doc_franky_Errors =
    R"doc(franka::Errors with noexcept copy and assignment.

franka::Errors only holds booleans, so copying it cannot throw, but libfranka does not mark its
special members noexcept. This wrapper does, keeping RobotState nothrow move constructible so it can
be captured in real-time queued callbacks (see RTFunctionQueue).)doc";

static const char *mkd_doc_franky_Errors_Errors = R"doc()doc";

static const char *mkd_doc_franky_Errors_Errors_2 = R"doc()doc";

static const char *mkd_doc_franky_Errors_Errors_3 = R"doc()doc";

static const char *mkd_doc_franky_Errors_operator_assign = R"doc()doc";

static const char *mkd_doc_franky_FlipDirection = R"doc(Flip direction of a joint.)doc";

static const char *mkd_doc_franky_FlipDirection_kNegative = R"doc(The joint angle is negative.)doc";

static const char *mkd_doc_franky_FlipDirection_kNeutral = R"doc(The joint angle is zero.)doc";

static const char *mkd_doc_franky_FlipDirection_kPositive = R"doc(The joint angle is positive.)doc";

static const char *mkd_doc_franky_FrictionCompensationParams =
    R"doc(Per-joint friction feedforward settings for torque-control motions.)doc";

static const char *mkd_doc_franky_FrictionCompensationParams_FrictionCompensationParams = R"doc()doc";

static const char *mkd_doc_franky_FrictionCompensationParams_FrictionCompensationParams_2 = R"doc()doc";

static const char *mkd_doc_franky_FrictionCompensationParams_coulomb =
    R"doc(Coulomb friction compensation gains in [Nm].)doc";

static const char *mkd_doc_franky_FrictionCompensationParams_max_torque =
    R"doc(Absolute per-joint clamp for friction compensation in [Nm].)doc";

static const char *mkd_doc_franky_FrictionCompensationParams_validate =
    R"doc(Throw std::invalid_argument if any parameter is out of range.

Fields are mutable, so consumers of this struct call this again at the point of use.

)doc";

static const char *mkd_doc_franky_FrictionCompensationParams_velocity_epsilon =
    R"doc(Velocity scale for the smooth Coulomb sign transition in [rad/s].)doc";

static const char *mkd_doc_franky_FrictionCompensationParams_viscous =
    R"doc(Viscous friction compensation gains in [Nms/rad].)doc";

static const char *mkd_doc_franky_Gripper =
    R"doc(A wrapper around the franka::Gripper class that adds asynchronous functionality.)doc";

static const char *mkd_doc_franky_GripperException = R"doc(Exception thrown by the gripper class.)doc";

static const char *mkd_doc_franky_Gripper_Gripper = R"doc()doc";

static const char *mkd_doc_franky_Gripper_Gripper_2 = R"doc()doc";

static const char *mkd_doc_franky_Gripper_current_future = R"doc()doc";

static const char *mkd_doc_franky_Gripper_graspAsync =
    R"doc(Asynchronous variant of the grasp function.

Args:
    width: Size of the object to grasp in [m].
    speed: Closing speed in [m/s].
    force: Grasping force in [N].
    epsilon_inner: Maximum tolerated deviation when the actual grasped width is smaller than the
                   commanded grasp width.
    epsilon_outer: Maximum tolerated deviation when the actual grasped width is larger than the
                   commanded grasp width.

Returns:
    Future that becomes ready when the grasp is finished. Contains true if an object has been
    grasped, false otherwise.

)doc";

static const char *mkd_doc_franky_Gripper_homingAsync =
    R"doc(Asynchronous variant of the homing function.

Returns:
    A future that becomes ready when the homing is finished. Contains true if the homing was
    successful.

)doc";

static const char *mkd_doc_franky_Gripper_is_grasped = R"doc(Whether the gripper is grasping)doc";

static const char *mkd_doc_franky_Gripper_max_width = R"doc(Maximum width of the gripper [m])doc";

static const char *mkd_doc_franky_Gripper_moveAsync =
    R"doc(Asynchronous variant of the move function.

Args:
    width: Intended opening width in [m].
    speed: Speed of the movement in [m/s].

Returns:
    Future that becomes ready when the movement is finished. Contains true if the movement was
    successful.

)doc";

static const char *mkd_doc_franky_Gripper_open =
    R"doc(Opens the gripper fully.

Args:
    speed: Speed of the movement in [m/s].

Returns:
    True if the gripper was opened successfully.

)doc";

static const char *mkd_doc_franky_Gripper_openAsync =
    R"doc(Asynchronous variant of the open function.

Args:
    speed: Speed of the movement in [m/s].

Returns:
    Future that becomes ready when the gripper is fully opened. Contains true if the gripper was
    opened successfully.

)doc";

static const char *mkd_doc_franky_Gripper_setCurrentFuture = R"doc()doc";

static const char *mkd_doc_franky_Gripper_state = R"doc(Current gripper state.)doc";

static const char *mkd_doc_franky_Gripper_stopAsync =
    R"doc(Asynchronous variant of the stop function.

Returns:
    A future that becomes ready when the stop is finished. Contains true if the stop was successful.

)doc";

static const char *mkd_doc_franky_Gripper_width = R"doc(Current opening width of the gripper [m])doc";

static const char *mkd_doc_franky_InvalidMotionTypeException =
    R"doc(Exception thrown when an invalid motion type is used.

This exception is thrown when a motion is asynchronously executed and a new motion of a different
type is set before the previous one finished.)doc";

static const char *mkd_doc_franky_JointImpedanceBase =
    R"doc(Base class for client-side joint impedance motions.

This class computes joint torques from a joint-space spring-damper law plus optional torque
offset/model compensation. Subclasses implement nextCommandImpl and call computeCommand with their
current reference.)doc";

static const char *mkd_doc_franky_JointImpedanceBase_CartesianShapingState = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceBase_CartesianShapingState_critical_damping = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceBase_CartesianShapingState_critical_damping_stiffness = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceBase_CartesianShapingState_damping = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceBase_CartesianShapingState_stiffness = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceBase_JointImpedanceBase = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceBase_cartesian_gains_handle = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceBase_cartesian_shaping = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceBase_computeCommand = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceBase_criticalShapingDamping = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceBase_current_damping = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceBase_current_stiffness = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceBase_gains_handle = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceBase_gains_time_constant = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceBase_getCartesianGains =
    R"doc(Get a copy of the current target Cartesian shaping gains.

Mutating the returned object has no effect on the motion; pass it to setCartesianGains to apply
changes.

)doc";

static const char *mkd_doc_franky_JointImpedanceBase_getGains =
    R"doc(Get a copy of the current target impedance gains.

Mutating the returned object has no effect on the motion; pass it to setGains to apply changes.

)doc";

static const char *mkd_doc_franky_JointImpedanceBase_params = R"doc(The parameters of the motion.)doc";

static const char *mkd_doc_franky_JointImpedanceBase_params_2 = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceBase_setCartesianGains =
    R"doc(Set the target Cartesian shaping gains.

The gains are validated and then smoothed in the control loop via exponential interpolation.

Args:
    gains: The new target Cartesian gains.

)doc";

static const char *mkd_doc_franky_JointImpedanceBase_setGains =
    R"doc(Set the target impedance gains.

The gains are validated and then smoothed in the control loop via exponential interpolation.

Args:
    gains: The new target gains.

)doc";

static const char *mkd_doc_franky_JointImpedanceBase_target =
    R"doc(The target joint positions of the motion [rad].)doc";

static const char *mkd_doc_franky_JointImpedanceBase_target_2 = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceBase_target_velocity =
    R"doc(The target joint velocities of the motion [rad/s].)doc";

static const char *mkd_doc_franky_JointImpedanceBase_target_velocity_2 = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceGains =
    R"doc(Runtime-adjustable stiffness and damping gains for joint impedance motions.)doc";

static const char *mkd_doc_franky_JointImpedanceGains_JointImpedanceGains = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceGains_JointImpedanceGains_2 = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceGains_damping =
    R"doc(Joint damping gains [Nms/rad]. If unset, the controller tracks critical damping, 2*sqrt(stiffness),
against the current (smoothed) stiffness every control cycle, so damping stays critical through gain
transitions.)doc";

static const char *mkd_doc_franky_JointImpedanceGains_stiffness = R"doc(Joint stiffness gains [Nm/rad].)doc";

static const char *mkd_doc_franky_JointImpedanceGains_validate =
    R"doc(Throw std::invalid_argument if any gain is negative or non-finite.)doc";

static const char *mkd_doc_franky_JointImpedanceMotion =
    R"doc(Client-side joint impedance controller.

This motion uses Franka's torque interface and regulates toward a constant joint target (and
optional target velocity). It does not terminate on its own; it runs until it is preempted or
stopped.)doc";

static const char *mkd_doc_franky_JointImpedanceMotion_JointImpedanceMotion = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceMotion_JointImpedanceMotion_2 = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceMotion_JointImpedanceMotion_3 = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceMotion_JointImpedanceMotion_4 = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceMotion_nextCommandImpl = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceParams = R"doc(Parameters for joint impedance motions.)doc";

static const char *mkd_doc_franky_JointImpedanceParams_cartesian_gains =
    R"doc(Optional Cartesian-space stiffness/damping projected through the current Jacobian.)doc";

static const char *mkd_doc_franky_JointImpedanceParams_compensate_coriolis =
    R"doc(Compensate Coriolis forces using the robot model.)doc";

static const char *mkd_doc_franky_JointImpedanceParams_constant_torque_offset =
    R"doc(Constant torque offset added to every command in [Nm].)doc";

static const char *mkd_doc_franky_JointImpedanceParams_damping =
    R"doc(Joint damping gains in [Nms/rad]. If unset, critical damping is tracked against the
 current (smoothed) stiffness.)doc";

static const char *mkd_doc_franky_JointImpedanceParams_error_clip =
    R"doc(Maximum absolute joint position error [rad] used by the joint-space controller.)doc";

static const char *mkd_doc_franky_JointImpedanceParams_friction = R"doc(Joint friction compensation settings.)doc";

static const char *mkd_doc_franky_JointImpedanceParams_safety =
    R"doc(Shared torque safety limits and soft joint-limit repulsion settings.)doc";

static const char *mkd_doc_franky_JointImpedanceParams_stiffness = R"doc(Joint stiffness gains in [Nm/rad].)doc";

static const char *mkd_doc_franky_JointImpedanceParams_validate =
    R"doc(Throw std::invalid_argument if any parameter is out of range.)doc";

static const char *mkd_doc_franky_JointImpedanceTrackingMotion =
    R"doc(Client-side joint impedance controller with a dynamic online reference.

This motion keeps the same controller alive while reading the latest valid joint reference from a
handle or callback each control cycle.)doc";

static const char *mkd_doc_franky_JointImpedanceTrackingMotion_JointImpedanceTrackingMotion = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceTrackingMotion_JointImpedanceTrackingMotion_2 = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceTrackingMotion_getReference =
    R"doc(Get a copy of the last commanded joint reference, or nullopt if no reference has been set yet.

Mutating the returned object has no effect on the motion; pass it to setReference to apply changes.

)doc";

static const char *mkd_doc_franky_JointImpedanceTrackingMotion_initImpl = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceTrackingMotion_nextCommandImpl = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceTrackingMotion_reference_callback = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceTrackingMotion_reference_handle = R"doc()doc";

static const char *mkd_doc_franky_JointImpedanceTrackingMotion_setReference =
    R"doc(Set the joint reference tracked by the controller.

The reference is validated and picked up by the control loop in the next cycle.

Args:
    reference: The new reference.

)doc";

static const char *mkd_doc_franky_JointMotion = R"doc(Joint motion with a single target.)doc";

static const char *mkd_doc_franky_JointMotion_JointMotion =
    R"doc(Args:
    target: The target joint state.
    reference_type: The reference type (absolute or relative). An absolute target is defined in the
                    robot's base frame, a relative target is defined relative to the current joint
                    position.
    relative_dynamics_factor: The relative dynamics factor for this motion. The factor will get
                              multiplied with the robot's global dynamics factor to get the actual
                              dynamics factor for this motion.
    return_when_finished: Whether to end the motion when the target is reached or keep holding the
                          last target.

)doc";

static const char *mkd_doc_franky_JointReference =
    R"doc(Joint-space reference for joint impedance motions.

The impedance controller tracks the joint position and velocity reference and adds the optional per-
cycle feedforward torque term to the commanded torques.)doc";

static const char *mkd_doc_franky_JointReference_dq = R"doc(Desired joint velocities [rad/s].)doc";

static const char *mkd_doc_franky_JointReference_q = R"doc(Desired joint positions [rad].)doc";

static const char *mkd_doc_franky_JointReference_tau_ff =
    R"doc(Feedforward torque added to the commanded torques [Nm].)doc";

static const char *mkd_doc_franky_JointReference_validate =
    R"doc(Throw std::invalid_argument if any value is non-finite.)doc";

static const char *mkd_doc_franky_JointState =
    R"doc(Joint state of a robot.

This class encapsulates the joint state of a robot, which comprises the joint positions and the
joint velocities.)doc";

static const char *mkd_doc_franky_JointState_JointState =
    R"doc(Construct a joint state with the given joint positions and zero velocities.

Args:
    position: The joint positions.

)doc";

static const char *mkd_doc_franky_JointState_JointState_2 =
    R"doc(Args:
    position: The joint positions.
    velocity: The joint velocities.

)doc";

static const char *mkd_doc_franky_JointState_JointState_3 = R"doc()doc";

static const char *mkd_doc_franky_JointState_JointState_4 = R"doc()doc";

static const char *mkd_doc_franky_JointState_position = R"doc(The position component of the state.)doc";

static const char *mkd_doc_franky_JointState_position_2 = R"doc()doc";

static const char *mkd_doc_franky_JointState_velocity = R"doc(The velocity component of the state.)doc";

static const char *mkd_doc_franky_JointState_velocity_2 = R"doc()doc";

static const char *mkd_doc_franky_JointVelocityMotion = R"doc(Joint velocity motion with a single target.)doc";

static const char *mkd_doc_franky_JointVelocityMotion_JointVelocityMotion =
    R"doc(Args:
    target: The target joint velocity.
    duration: For how long the target is held after it has been reached. Default is 1s.
    relative_dynamics_factor: The relative dynamics factor for this motion. The factor will get
                              multiplied with the robot's global dynamics factor to get the actual
                              dynamics factor for this motion.

)doc";

static const char *mkd_doc_franky_JointVelocityWaypointMotion =
    R"doc(Joint velocity waypoint motion.

This motion follows multiple joint velocity waypoints in a time-optimal way.)doc";

static const char *mkd_doc_franky_JointVelocityWaypointMotion_JointVelocityWaypointMotion =
    R"doc(Args:
    waypoints: Joint waypoints to follow.
    relative_dynamics_factor: The relative dynamics factor for this motion. The factor will get
                              multiplied with the robot's global dynamics factor to get the actual
                              dynamics factor for this motion.

)doc";

static const char *mkd_doc_franky_JointVelocityWaypointMotion_checkWaypoint = R"doc()doc";

static const char *mkd_doc_franky_JointVelocityWaypointMotion_getAbsoluteInputLimits = R"doc()doc";

static const char *mkd_doc_franky_JointVelocityWaypointMotion_getControlSignal = R"doc()doc";

static const char *mkd_doc_franky_JointVelocityWaypointMotion_getDesiredState = R"doc()doc";

static const char *mkd_doc_franky_JointVelocityWaypointMotion_initWaypointMotion = R"doc()doc";

static const char *mkd_doc_franky_JointVelocityWaypointMotion_setNewWaypoint = R"doc()doc";

static const char *mkd_doc_franky_JointWaypointMotion =
    R"doc(Joint waypoint motion.

This motion follows multiple joint waypoints in a time-optimal way.)doc";

static const char *mkd_doc_franky_JointWaypointMotion_JointWaypointMotion =
    R"doc(Args:
    waypoints: Joint waypoints to follow.
    relative_dynamics_factor: The relative dynamics factor for this motion. The factor will get
                              multiplied with the robot's global dynamics factor to get the actual
                              dynamics factor for this motion.
    return_when_finished: Whether to end the motion when the last waypoint is reached or keep
                          holding the last target.

)doc";

static const char *mkd_doc_franky_JointWaypointMotion_getAbsoluteInputLimits = R"doc()doc";

static const char *mkd_doc_franky_JointWaypointMotion_getControlSignal = R"doc()doc";

static const char *mkd_doc_franky_JointWaypointMotion_getDesiredState = R"doc()doc";

static const char *mkd_doc_franky_JointWaypointMotion_initWaypointMotion = R"doc()doc";

static const char *mkd_doc_franky_JointWaypointMotion_setNewWaypoint = R"doc()doc";

static const char *mkd_doc_franky_JointWaypointMotion_target_position = R"doc()doc";

static const char *mkd_doc_franky_ManipulabilityTask =
    R"doc(Manipulability maximization objective projected into the Cartesian nullspace.)doc";

static const char *mkd_doc_franky_ManipulabilityTask_ManipulabilityTask = R"doc()doc";

static const char *mkd_doc_franky_ManipulabilityTask_ManipulabilityTask_2 = R"doc()doc";

static const char *mkd_doc_franky_ManipulabilityTask_damping =
    R"doc(Joint damping applied to this task before projection [Nms/rad].)doc";

static const char *mkd_doc_franky_ManipulabilityTask_gain = R"doc(Gain applied to the manipulability gradient.)doc";

static const char *mkd_doc_franky_ManipulabilityTask_max_torque =
    R"doc(Per-joint absolute torque clamp for this task [Nm]. Unset means no clamp.)doc";

static const char *mkd_doc_franky_ManipulabilityTask_validate =
    R"doc(Throw std::invalid_argument if any task parameter is out of range.)doc";

static const char *mkd_doc_franky_Measure =
    R"doc(A measure on the robot state.

This class defines a measure on the robot state, which can be used to define a condition for a
reaction in a motion. Measures support arithmetic operations (``+``, ``-``, ``*``, ``/``, ``^``) and
comparisons (``==``, ``!=``, ``<``, ``>``, ``<=``, ``>=``) can be combined to form more complex
measures.)doc";

static const char *mkd_doc_franky_Measure_AbsTime =
    R"doc(A measure that returns the absolute time since the start of the current chain of motions. The
absolute time measures the time since the robot started moving, and is only reset if a motion
expires without being replaced by a new motion.

)doc";

static const char *mkd_doc_franky_Measure_ForceX =
    R"doc(A measure that returns the linear force on the end-effector in X direction as by the O_F_ext_hat_K
component of the robot state.

)doc";

static const char *mkd_doc_franky_Measure_ForceY =
    R"doc(A measure that returns the linear force on the end-effector in Y direction as by the O_F_ext_hat_K
component of the robot state.

)doc";

static const char *mkd_doc_franky_Measure_ForceZ =
    R"doc(A measure that returns the linear force on the end-effector in Z direction as by the O_F_ext_hat_K
component of the robot state.

)doc";

static const char *mkd_doc_franky_Measure_Measure =
    R"doc(Args:
    measure_func: A function that returns the value of the measure.
    repr: A string representation of the measure.

)doc";

static const char *mkd_doc_franky_Measure_Measure_2 =
    R"doc(Implicit constructor for constant measures.

Args:
    constant: The constant value of the measure.

)doc";

static const char *mkd_doc_franky_Measure_RelTime =
    R"doc(A measure that returns the relative time since the start of the current motion.)doc";

static const char *mkd_doc_franky_Measure_measure_func = R"doc()doc";

static const char *mkd_doc_franky_Measure_operator_call =
    R"doc(Get the value of the measure.

Args:
    robot_state: The current robot state.
    rel_time: The time since the start of the current motion.
    abs_time: The time since the start of the current chain of motions. This value measures the time
              since the robot started moving, and is only reset if a motion expires without being
              replaced by a new motion.

Returns:
    The value of the measure.

)doc";

static const char *mkd_doc_franky_Measure_repr = R"doc(The string representation of the measure.)doc";

static const char *mkd_doc_franky_Measure_repr_2 = R"doc()doc";

static const char *mkd_doc_franky_Model =
    R"doc(A wrapper around franka::Model that uses Eigen types.

This class exposes the same functionality as franka::Model, but uses Eigen types for inputs and
outputs instead of std::array. All 2D arrays are returned as Eigen matrices.)doc";

static const char *mkd_doc_franky_Model_Model =
    R"doc(Args:
    model: The underlying franka::Model instance to wrap.

)doc";

static const char *mkd_doc_franky_Model_Model_2 = R"doc()doc";

static const char *mkd_doc_franky_Model_Model_3 = R"doc()doc";

static const char *mkd_doc_franky_Model_bodyJacobian =
    R"doc(Calculates the body Jacobian in the given frame.

Args:
    frame: The frame for which the Jacobian is computed.
    state: The current robot state.

Returns:
    The 6x7 body Jacobian matrix.

)doc";

static const char *mkd_doc_franky_Model_bodyJacobian_2 =
    R"doc(Calculates the body Jacobian in the given frame.

Args:
    frame: The frame for which the Jacobian is computed.
    q: Robot joint angles [rad].
    F_T_EE: Transformation from flange to end-effector frame.
    EE_T_K: Transformation from end-effector frame to stiffness frame.

Returns:
    The 6x7 body Jacobian matrix.

)doc";

static const char *mkd_doc_franky_Model_coriolis =
    R"doc(Calculates the Coriolis force vector.

Args:
    state: The current robot state.

Returns:
    The Coriolis vector [Nm].

)doc";

static const char *mkd_doc_franky_Model_coriolis_2 =
    R"doc(Calculates the Coriolis force vector.

Args:
    q: Robot joint angles [rad].
    dq: Robot joint velocities [rad/s].
    I_total: Combined load and robot inertia [kg·m²].
    m_total: Combined mass of robot and load [kg].
    F_x_Ctotal: Center of mass relative to flange frame [m].

Returns:
    The Coriolis vector [Nm].

)doc";

static const char *mkd_doc_franky_Model_gravity =
    R"doc(Calculates the gravity vector.

Args:
    state: The current robot state.
    gravity_earth: Gravity vector in base frame [m/s²].

Returns:
    The gravity vector [Nm].

)doc";

static const char *mkd_doc_franky_Model_gravity_2 =
    R"doc(Calculates the gravity vector using default gravity direction (0, 0, -9.81).

Args:
    state: The current robot state.

Returns:
    The gravity vector [Nm].

)doc";

static const char *mkd_doc_franky_Model_gravity_3 =
    R"doc(Calculates the gravity vector.

Args:
    q: Robot joint angles [rad].
    m_total: Combined mass of robot and load [kg].
    F_x_Ctotal: Center of mass relative to flange frame [m].
    gravity_earth: Gravity vector in base frame [m/s²], default is (0, 0, -9.81).

Returns:
    The gravity vector [Nm].

)doc";

static const char *mkd_doc_franky_Model_mass =
    R"doc(Calculates the mass matrix.

Args:
    state: The current robot state.

Returns:
    The 7x7 mass matrix.

)doc";

static const char *mkd_doc_franky_Model_mass_2 =
    R"doc(Calculates the mass matrix.

Args:
    q: Robot joint angles [rad].
    I_total: Combined load and robot inertia [kg·m²].
    m_total: Combined mass of robot and load [kg].
    F_x_Ctotal: Center of mass relative to flange frame [m].

Returns:
    The 7x7 mass matrix.

)doc";

static const char *mkd_doc_franky_Model_model = R"doc()doc";

static const char *mkd_doc_franky_Model_operator_assign = R"doc()doc";

static const char *mkd_doc_franky_Model_operator_assign_2 = R"doc()doc";

static const char *mkd_doc_franky_Model_pose =
    R"doc(Calculates the pose of a frame relative to the base frame.

Args:
    frame: The frame whose pose should be returned.
    state: The current robot state.

Returns:
    The pose as an affine transformation matrix.

)doc";

static const char *mkd_doc_franky_Model_pose_2 =
    R"doc(Calculates the pose of a frame relative to the base frame.

Args:
    frame: The frame whose pose should be returned.
    q: Robot joint angles [rad].
    F_T_EE: Transformation from flange to end-effector frame.
    EE_T_K: Transformation from end-effector frame to stiffness frame.

Returns:
    The pose as an affine transformation matrix.

)doc";

static const char *mkd_doc_franky_Model_zeroJacobian =
    R"doc(Calculates the zero Jacobian in base frame.

Args:
    frame: The frame for which the Jacobian is computed.
    state: The current robot state.

Returns:
    The 6x7 zero Jacobian matrix.

)doc";

static const char *mkd_doc_franky_Model_zeroJacobian_2 =
    R"doc(Calculates the zero Jacobian in base frame.

Args:
    frame: The frame for which the Jacobian is computed.
    q: Robot joint angles [rad].
    F_T_EE: Transformation from flange to end-effector frame.
    EE_T_K: Transformation from end-effector frame to stiffness frame.

Returns:
    The 6x7 zero Jacobian matrix.

)doc";

static const char *mkd_doc_franky_Motion =
    R"doc(Base class for motions.

Template Args:
    ControlSignalType: Control signal type of the motion. Either franka::Torques,
                       franka::JointVelocities, franka::CartesianVelocities, franka::JointPositions
                       or franka::CartesianPose.)doc";

static const char *mkd_doc_franky_Motion_2 = R"doc()doc";

static const char *mkd_doc_franky_Motion_3 =
    R"doc(Base class for motions.

Template Args:
    ControlSignalType: Control signal type of the motion. Either franka::Torques,
                       franka::JointVelocities, franka::CartesianVelocities, franka::JointPositions
                       or franka::CartesianPose.)doc";

static const char *mkd_doc_franky_Motion_4 =
    R"doc(Base class for motions.

Template Args:
    ControlSignalType: Control signal type of the motion. Either franka::Torques,
                       franka::JointVelocities, franka::CartesianVelocities, franka::JointPositions
                       or franka::CartesianPose.)doc";

static const char *mkd_doc_franky_MotionGenerator = R"doc(Helper class for handling motions and reactions.)doc";

static const char *mkd_doc_franky_MotionGenerator_MotionGenerator =
    R"doc(Args:
    robot: The robot instance.
    initial_motion: The initial motion.

)doc";

static const char *mkd_doc_franky_MotionGenerator_abs_time = R"doc()doc";

static const char *mkd_doc_franky_MotionGenerator_current_motion = R"doc()doc";

static const char *mkd_doc_franky_MotionGenerator_has_new_motion = R"doc(Whether a new motion is available.)doc";

static const char *mkd_doc_franky_MotionGenerator_initial_motion = R"doc()doc";

static const char *mkd_doc_franky_MotionGenerator_new_motion = R"doc()doc";

static const char *mkd_doc_franky_MotionGenerator_new_motion_mutex = R"doc()doc";

static const char *mkd_doc_franky_MotionGenerator_operator_call =
    R"doc(Update the motion generator and get the next control signal.

Args:
    robot_state: The current robot state.
    period: The time step.

Returns:
    The control signal for the robot.

)doc";

static const char *mkd_doc_franky_MotionGenerator_previous_command = R"doc()doc";

static const char *mkd_doc_franky_MotionGenerator_registerUpdateCallback =
    R"doc(Register a callback that is called in every step of the motion.

Args:
    callback: The callback to register. Callbacks are called with the robot state, the time step
              [s], the relative time [s] and the control signal computed in this step.

)doc";

static const char *mkd_doc_franky_MotionGenerator_rel_time_offset = R"doc()doc";

static const char *mkd_doc_franky_MotionGenerator_resetTimeUnsafe =
    R"doc(Reset the time of the motion generator without locking the mutex.)doc";

static const char *mkd_doc_franky_MotionGenerator_robot = R"doc()doc";

static const char *mkd_doc_franky_MotionGenerator_updateMotion =
    R"doc(Update the current motion.

Args:
    new_motion: The new motion.

)doc";

static const char *mkd_doc_franky_MotionGenerator_update_callbacks = R"doc()doc";

static const char *mkd_doc_franky_MotionPlannerException = R"doc(Exception thrown if the motion planner fails.)doc";

static const char *mkd_doc_franky_MotionReuseException =
    R"doc(Exception thrown when a motion is started more than once.

Motions maintain internal state, hence they must not be reused. Create a new motion instance for
every execution instead.)doc";

static const char *mkd_doc_franky_Motion_Motion = R"doc()doc";

static const char *mkd_doc_franky_Motion_addReaction =
    R"doc(Add a reaction to the motion.

Reactions are evaluated in every step of the motion and can replace the current motion with a new
motion.

Args:
    reaction: The reaction to add.

Warning:
    Do not call this function synchronously from a condition, motion function, or reaction callback
    while this motion's reactions are being evaluated. Reaction evaluation holds the reaction-list
    mutex, so doing so would deadlock.

)doc";

static const char *mkd_doc_franky_Motion_addReactionFront =
    R"doc(Add a reaction to the front of the reaction list.

Reactions are evaluated in every step of the motion and can replace the current motion with a new
motion.

Args:
    reaction: The reaction to add.

Warning:
    Do not call this function synchronously from a condition, motion function, or reaction callback
    while this motion's reactions are being evaluated. Reaction evaluation holds the reaction-list
    mutex, so doing so would deadlock.

)doc";

static const char *mkd_doc_franky_Motion_callback_mutex = R"doc()doc";

static const char *mkd_doc_franky_Motion_callbacks = R"doc()doc";

static const char *mkd_doc_franky_Motion_checkAndCallReactions =
    R"doc(Check and call reactions.

Args:
    robot_state: The current robot state.
    rel_time: The relative time.
    abs_time: The absolute time [s].

Returns:
    The new motion if a reaction was triggered, nullptr otherwise.

)doc";

static const char *mkd_doc_franky_Motion_has_started =
    R"doc(Whether this motion has already been started. Started motions cannot be started again.)doc";

static const char *mkd_doc_franky_Motion_init =
    R"doc(Initialize the motion. Motions can only be started once; this function throws a MotionReuseException
on any subsequent call.

Args:
    robot: The robot instance.
    robot_state: The current robot state.
    previous_command: The previous command.

)doc";

static const char *mkd_doc_franky_Motion_initImpl = R"doc()doc";

static const char *mkd_doc_franky_Motion_nextCommand =
    R"doc(Get the next command of the motion.

Args:
    robot_state: The current robot state.
    time_step: The time step [s].
    rel_time: The relative time.
    abs_time: The absolute time [s].
    previous_command: The previous command.

Returns:
    The next control signal for libfranka.

)doc";

static const char *mkd_doc_franky_Motion_nextCommandImpl = R"doc()doc";

static const char *mkd_doc_franky_Motion_reaction_mutex = R"doc()doc";

static const char *mkd_doc_franky_Motion_reactions = R"doc(Currently registered reactions of the motion.)doc";

static const char *mkd_doc_franky_Motion_reactions_2 = R"doc()doc";

static const char *mkd_doc_franky_Motion_registerCallback =
    R"doc(Register a callback that is called in every step of the motion.

Args:
    callback: The callback to register. Callbacks are called with the robot state, the time step
              [s], the relative time [s], the absolute time [s] and the control signal computed in
              this step.

)doc";

static const char *mkd_doc_franky_Motion_robot = R"doc()doc";

static const char *mkd_doc_franky_Motion_robot_2 = R"doc()doc";

static const char *mkd_doc_franky_Motion_started = R"doc()doc";

static const char *mkd_doc_franky_NullspaceGains = R"doc(Runtime-adjustable gains for a nullspace task.)doc";

static const char *mkd_doc_franky_NullspaceGains_manipulability_damping =
    R"doc(Joint damping applied to the manipulability task before projection [Nms/rad].)doc";

static const char *mkd_doc_franky_NullspaceGains_manipulability_gain =
    R"doc(Gain applied to the manipulability gradient.)doc";

static const char *mkd_doc_franky_NullspaceGains_manipulability_max_torque =
    R"doc(Absolute torque clamp for the manipulability task [Nm]. Unset means no clamp.)doc";

static const char *mkd_doc_franky_NullspaceGains_posture_damping =
    R"doc(Per-joint posture damping [Nms/rad]. If unset, critical damping is used.)doc";

static const char *mkd_doc_franky_NullspaceGains_posture_max_torque =
    R"doc(Absolute torque clamp for the posture task [Nm]. Unset means no clamp.)doc";

static const char *mkd_doc_franky_NullspaceGains_posture_stiffness =
    R"doc(Per-joint posture stiffness [Nm/rad]. A joint with zero stiffness is not pushed.)doc";

static const char *mkd_doc_franky_NullspaceGains_validate =
    R"doc(Throw std::invalid_argument if any gain or clamp is out of range.)doc";

static const char *mkd_doc_franky_PositionWaypoint =
    R"doc(A position waypoint with a target and optional parameters.

Template Args:
    TargetType: The type of the target.)doc";

static const char *mkd_doc_franky_PositionWaypointMotion =
    R"doc(A motion following multiple positional waypoints in a time-optimal way. Works with arbitrary initial
conditions.

Template Args:
    ControlSignalType: The type of the control signal. Either franka::Torques,
                       franka::JointVelocities, franka::CartesianVelocities, franka::JointPositions
                       or franka::CartesianPose.
    TargetType: The type of the target of the waypoints.)doc";

static const char *mkd_doc_franky_PositionWaypointMotion_PositionWaypointMotion =
    R"doc(Args:
    waypoints: The waypoints to follow.
    relative_dynamics_factor: The relative dynamics factor for this motion. This factor will get
                              multiplied with the robot's global dynamics factor to get the actual
                              dynamics factor for this motion.
    return_when_finished: Whether to end the motion when the last waypoint is reached or keep
                          holding the last target.

)doc";

static const char *mkd_doc_franky_PositionWaypointMotion_extrapolateMotion = R"doc()doc";

static const char *mkd_doc_franky_PositionWaypointMotion_getAbsoluteInputLimits = R"doc()doc";

static const char *mkd_doc_franky_PositionWaypointMotion_getDesiredState = R"doc()doc";

static const char *mkd_doc_franky_PositionWaypointMotion_relative_dynamics_factor =
    R"doc(The relative dynamics factor of this motion.)doc";

static const char *mkd_doc_franky_PositionWaypointMotion_relative_dynamics_factor_2 = R"doc()doc";

static const char *mkd_doc_franky_PositionWaypointMotion_setInputLimits = R"doc()doc";

static const char *mkd_doc_franky_PositionWaypoint_reference_type =
    R"doc(The reference type (absolute or relative). A relative waypoint is resolved against the preceding
waypoint's target. The first relative waypoint is resolved against the motion's initial commanded
position. This remains true if max_total_duration advances past a waypoint before it is reached.)doc";

static const char *mkd_doc_franky_PostureTask =
    R"doc(Joint-posture objective projected into the Cartesian nullspace.)doc";

static const char *mkd_doc_franky_PostureTask_PostureTask = R"doc()doc";

static const char *mkd_doc_franky_PostureTask_PostureTask_2 = R"doc()doc";

static const char *mkd_doc_franky_PostureTask_PostureTask_3 =
    R"doc(Convenience constructor applying the same scalar gains to all joints.)doc";

static const char *mkd_doc_franky_PostureTask_damping =
    R"doc(Per-joint posture damping in [Nms/rad].

If unset, the controller uses critical damping, 2*sqrt(stiffness), per joint.)doc";

static const char *mkd_doc_franky_PostureTask_max_torque =
    R"doc(Per-joint absolute torque clamp for this task [Nm]. Unset means no clamp.)doc";

static const char *mkd_doc_franky_PostureTask_stiffness =
    R"doc(Per-joint posture stiffness in [Nm/rad].

A joint with zero stiffness is not pushed by this task. Note that the task torque is projected into
the Jacobian nullspace, so pushing a single joint still moves all joints that participate in the
self-motion.)doc";

static const char *mkd_doc_franky_PostureTask_target = R"doc(Preferred joint posture [rad].)doc";

static const char *mkd_doc_franky_PostureTask_validate =
    R"doc(Throw std::invalid_argument if any task parameter is out of range.)doc";

static const char *mkd_doc_franky_Reaction =
    R"doc(A reaction that can be attached to a motion.

This class defines a reaction in a motion, which can be used to change the motion based on the robot
state. Reactions consist of a condition and a motion that replaces the current motion immediately if
the condition is met.)doc";

static const char *mkd_doc_franky_Reaction_2 =
    R"doc(A reaction that can be attached to a motion.

This class defines a reaction in a motion, which can be used to change the motion based on the robot
state. Reactions consist of a condition and a motion that replaces the current motion immediately if
the condition is met.)doc";

static const char *mkd_doc_franky_Reaction_3 =
    R"doc(A reaction that can be attached to a motion.

This class defines a reaction in a motion, which can be used to change the motion based on the robot
state. Reactions consist of a condition and a motion that replaces the current motion immediately if
the condition is met.)doc";

static const char *mkd_doc_franky_ReactionRecursionException =
    R"doc(Exception thrown when the reaction recursion limit (16) is reached.)doc";

static const char *mkd_doc_franky_Reaction_Reaction =
    R"doc(Args:
    condition: The condition that must be met for the reaction to be executed.
    new_motion: The motion that is executed if the condition is met.

)doc";

static const char *mkd_doc_franky_Reaction_Reaction_2 =
    R"doc(Args:
    condition: The condition that must be met for the reaction to be executed.
    motion_func: A function that returns a motion that is executed if the condition is met.

)doc";

static const char *mkd_doc_franky_Reaction_callback_mutex = R"doc()doc";

static const char *mkd_doc_franky_Reaction_callbacks = R"doc()doc";

static const char *mkd_doc_franky_Reaction_condition =
    R"doc(Check if the condition is met.

Args:
    robot_state: The current robot state.
    rel_time: The time since the start of the current motion.
    abs_time: The time since the start of the current chain of motions. This value measures the time
              since the robot started moving, and is only reset if a motion expires without being
              replaced by a new motion.

Returns:
    True if the condition is met, false otherwise.

)doc";

static const char *mkd_doc_franky_Reaction_condition_2 = R"doc()doc";

static const char *mkd_doc_franky_Reaction_motion_func = R"doc()doc";

static const char *mkd_doc_franky_Reaction_operator_call =
    R"doc(Get the new motion if the condition is met.

Args:
    robot_state: The current robot state.
    rel_time: The time since the start of the current motion.
    abs_time: The time since the start of the current chain of motions. This value measures the time
              since the robot started moving, and is only reset if a motion expires without being
              replaced by a new motion.

Returns:
    The new motion if the condition is met, or nullptr otherwise.

)doc";

static const char *mkd_doc_franky_Reaction_registerCallback =
    R"doc(Register a callback that is called when the condition of this reaction is met.

Args:
    callback: The callback to register. Callbacks are called with the robot state, the relative time
              [s] and the absolute time [s].

)doc";

static const char *mkd_doc_franky_ReferenceType =
    R"doc(Enum class for reference types.

This enum class defines the reference types for motions (absolute or relative).)doc";

static const char *mkd_doc_franky_ReferenceType_kAbsolute =
    R"doc(The target is given in the robot's base frame (or as absolute joint positions).)doc";

static const char *mkd_doc_franky_ReferenceType_kRelative =
    R"doc(The target is given relative to the robot's state when the motion starts.)doc";

static const char *mkd_doc_franky_RelativeDynamicsFactor =
    R"doc(Relative dynamics factors

This class encapsulates the relative dynamics factors, which are used to scale the maximum velocity,
acceleration, and jerk of a trajectory.)doc";

static const char *mkd_doc_franky_RelativeDynamicsFactor_MAX_DYNAMICS =
    R"doc(Special factor which causes the maximum dynamics to be used, independent of other factors applied
elsewhere.

)doc";

static const char *mkd_doc_franky_RelativeDynamicsFactor_RelativeDynamicsFactor =
    R"doc(Default constructor which initializes all values to 1.0.)doc";

static const char *mkd_doc_franky_RelativeDynamicsFactor_RelativeDynamicsFactor_2 =
    R"doc(Constructor which initializes all values to the given value.

Args:
    value: The value to initialize all factors with. Must be in the range (0, 1].

)doc";

static const char *mkd_doc_franky_RelativeDynamicsFactor_RelativeDynamicsFactor_3 =
    R"doc(Args:
    velocity: The factor for the velocity. Must be in the range (0, 1].
    acceleration: The factor for the acceleration. Must be in the range (0, 1].
    jerk: The factor for the jerk. Must be in the range (0, 1].

)doc";

static const char *mkd_doc_franky_RelativeDynamicsFactor_RelativeDynamicsFactor_4 = R"doc()doc";

static const char *mkd_doc_franky_RelativeDynamicsFactor_acceleration = R"doc(Acceleration factor.)doc";

static const char *mkd_doc_franky_RelativeDynamicsFactor_acceleration_2 = R"doc()doc";

static const char *mkd_doc_franky_RelativeDynamicsFactor_checkInBounds = R"doc()doc";

static const char *mkd_doc_franky_RelativeDynamicsFactor_jerk = R"doc(Jerk factor.)doc";

static const char *mkd_doc_franky_RelativeDynamicsFactor_jerk_2 = R"doc()doc";

static const char *mkd_doc_franky_RelativeDynamicsFactor_max_dynamics =
    R"doc(Whether the maximum dynamics should be used.)doc";

static const char *mkd_doc_franky_RelativeDynamicsFactor_max_dynamics_2 = R"doc()doc";

static const char *mkd_doc_franky_RelativeDynamicsFactor_operator_mul = R"doc()doc";

static const char *mkd_doc_franky_RelativeDynamicsFactor_velocity = R"doc(Velocity factor.)doc";

static const char *mkd_doc_franky_RelativeDynamicsFactor_velocity_2 = R"doc()doc";

static const char *mkd_doc_franky_Robot =
    R"doc(A class representing a Franka robot.

This class extends the franka::Robot class and adds additional functionality to it.)doc";

static const char *mkd_doc_franky_Robot_2 =
    R"doc(A class representing a Franka robot.

This class extends the franka::Robot class and adds additional functionality to it.)doc";

static const char *mkd_doc_franky_Robot_3 = R"doc()doc";

static const char *mkd_doc_franky_Robot_4 =
    R"doc(A class representing a Franka robot.

This class extends the franka::Robot class and adds additional functionality to it.)doc";

static const char *mkd_doc_franky_RobotPose =
    R"doc(Cartesian pose of a robot.

This class encapsulates the cartesian pose of a robot, which comprises the end effector pose and the
elbow position.)doc";

static const char *mkd_doc_franky_RobotPose_RobotPose = R"doc()doc";

static const char *mkd_doc_franky_RobotPose_RobotPose_2 = R"doc()doc";

static const char *mkd_doc_franky_RobotPose_RobotPose_3 =
    R"doc(Args:
    end_effector_pose: The pose of the end effector.
    elbow_state: The state of the elbow. Optional.

)doc";

static const char *mkd_doc_franky_RobotPose_RobotPose_4 =
    R"doc(Args:
    vector_repr: The vector representation of the pose.
    ignore_elbow: Whether to ignore the elbow state. Default is false.
    flip_direction: The flip direction to use if the elbow is not ignored. Default is negative.

)doc";

static const char *mkd_doc_franky_RobotPose_RobotPose_5 =
    R"doc(Args:
    vector_repr: The vector representation of the pose.
    elbow_state: The state of the elbow. Optional.

)doc";

static const char *mkd_doc_franky_RobotPose_RobotPose_6 =
    R"doc(Args:
    franka_pose: The franka pose.

)doc";

static const char *mkd_doc_franky_RobotPose_as_franka_pose =
    R"doc(Convert this pose to a franka pose.

Args:
    default_elbow_flip_direction: The default flip direction to use if the elbow flip direction is
                                  not set.

Returns:
    The franka pose.

)doc";

static const char *mkd_doc_franky_RobotPose_changeEndEffectorFrame =
    R"doc(Change the frame of the end effector by applying a transformation from the right side. This is
equivalent to calling rightTransform(transform).

Args:
    transform: The transform to apply.

Returns:
    The robot pose with the new end effector frame.

)doc";

static const char *mkd_doc_franky_RobotPose_elbow_state =
    R"doc(Get the elbow state.

Returns:
    The elbow state.

)doc";

static const char *mkd_doc_franky_RobotPose_elbow_state_2 = R"doc()doc";

static const char *mkd_doc_franky_RobotPose_end_effector_pose =
    R"doc(Get the end effector pose.

Returns:
    The end effector pose.

)doc";

static const char *mkd_doc_franky_RobotPose_end_effector_pose_2 = R"doc()doc";

static const char *mkd_doc_franky_RobotPose_leftTransform =
    R"doc(Transform this pose with a given affine transformation from the left side.

Args:
    transform: The transform to apply.

Returns:
    The transformed robot pose.

)doc";

static const char *mkd_doc_franky_RobotPose_rightTransform =
    R"doc(Transform this pose with a given affine transformation from the right side.

Args:
    transform: The transform to apply.

Returns:
    The transformed robot pose.

)doc";

static const char *mkd_doc_franky_RobotPose_vector_repr =
    R"doc(Get the vector representation of the pose, which consists of the end effector position and
orientation (as rotation vector) and the elbow position. Does not contain the flip component of the
elbow state.

Returns:
    The vector representation of the pose.

)doc";

static const char *mkd_doc_franky_RobotPose_withElbowState =
    R"doc(Get the pose with a new elbow state.

Args:
    elbow_state: The new elbow state.

Returns:
    The pose with the new elbow state.

)doc";

static const char *mkd_doc_franky_RobotState =
    R"doc(Full state of the robot

This class contains all fields of franka::RobotState and some additional fields. Each additional
field ends in "_est". Unlike franka::RobotState, all fields are converted to appropriate
Eigen/franky types.)doc";

static const char *mkd_doc_franky_RobotStateEstimator = R"doc()doc";

static const char *mkd_doc_franky_RobotStateEstimator_RobotStateEstimator = R"doc()doc";

static const char *mkd_doc_franky_RobotStateEstimator_RobotStateEstimator_2 = R"doc()doc";

static const char *mkd_doc_franky_RobotStateEstimator_control_adaptation_rate = R"doc()doc";

static const char *mkd_doc_franky_RobotStateEstimator_control_process_var = R"doc()doc";

static const char *mkd_doc_franky_RobotStateEstimator_ddq_process_var = R"doc()doc";

static const char *mkd_doc_franky_RobotStateEstimator_dq_process_var = R"doc()doc";

static const char *mkd_doc_franky_RobotStateEstimator_h_mat = R"doc()doc";

static const char *mkd_doc_franky_RobotStateEstimator_joint_state_covar = R"doc()doc";

static const char *mkd_doc_franky_RobotStateEstimator_joint_state_mean = R"doc()doc";

static const char *mkd_doc_franky_RobotStateEstimator_operator_call = R"doc()doc";

static const char *mkd_doc_franky_RobotStateEstimator_prev_time = R"doc()doc";

static const char *mkd_doc_franky_RobotStateEstimator_q_process_var = R"doc()doc";

static const char *mkd_doc_franky_RobotStateEstimator_r_mat = R"doc()doc";

static const char *mkd_doc_franky_RobotStateEstimator_update = R"doc()doc";

static const char *mkd_doc_franky_RobotState_EE_T_K =
    R"doc(:math:`^{EE}T_{K}` Stiffness frame pose in end effector frame.

See also the stiffness frame K.)doc";

static const char *mkd_doc_franky_RobotState_F_T_EE =
    R"doc(:math:`^{F}T_{EE}` End effector frame pose in flange frame.

See also:
    F_T_NE

See also:
    NE_T_EE

See also:
    Robot for an explanation of the F, NE and EE frames.)doc";

static const char *mkd_doc_franky_RobotState_F_T_NE =
    R"doc(:math:`^{F}T_{NE}` Nominal end effector frame pose in flange frame.

See also:
    F_T_EE

See also:
    NE_T_EE

See also:
    Robot for an explanation of the F, NE and EE frames.)doc";

static const char *mkd_doc_franky_RobotState_F_x_Cee =
    R"doc(:math:`^{F}x_{C_{EE}}` Configured center of mass of the end effector load with respect to flange
frame.)doc";

static const char *mkd_doc_franky_RobotState_F_x_Cload =
    R"doc(:math:`^{F}x_{C_{load}}` Configured center of mass of the external load with respect to flange
frame.)doc";

static const char *mkd_doc_franky_RobotState_F_x_Ctotal =
    R"doc(:math:`^{F}x_{C_{total}}` Combined center of mass of the end effector load and the external load
with respect to flange frame.)doc";

static const char *mkd_doc_franky_RobotState_I_ee =
    R"doc(:math:`I_{EE}` Configured rotational inertia matrix of the end effector load with respect to center
of mass.)doc";

static const char *mkd_doc_franky_RobotState_I_load =
    R"doc(:math:`I_{load}` Configured rotational inertia matrix of the external load with respect to center of
mass.)doc";

static const char *mkd_doc_franky_RobotState_I_total =
    R"doc(:math:`I_{total}` Combined rotational inertia matrix of the end effector load and the external load
with respect to the center of mass.)doc";

static const char *mkd_doc_franky_RobotState_K_F_ext_hat_K =
    R"doc(:math:`^{K}F_{K,\text{ext}}` Estimated external wrench (force, torque) acting on stiffness frame,
expressed relative to the stiffness frame. Forces applied by the robot to the environment are
positive, while forces applied by the environment on the robot are negative. Becomes
:math:`[0,0,0,0,0,0]` when near or in a singularity. See also the stiffness frame K. Unit:
:math:`[N,N,N,Nm,Nm,Nm]`.)doc";

static const char *mkd_doc_franky_RobotState_NE_T_EE =
    R"doc(:math:`^{NE}T_{EE}` End effector frame pose in nominal end effector frame.

See also:
    Robot::setEE to change this frame.

See also:
    F_T_EE

See also:
    F_T_NE

See also:
    Robot for an explanation of the F, NE and EE frames.)doc";

static const char *mkd_doc_franky_RobotState_O_F_ext_hat_K =
    R"doc(:math:`^OF_{K,\text{ext}}` Estimated external wrench (force, torque) acting on stiffness frame,
expressed relative to the the base frame. Forces applied by the robot to the environment are
positive, while forces applied by the environment on the robot are negative. Becomes
:math:`[0,0,0,0,0,0]` when near or in a singularity. See also the stiffness frame K. Unit:
:math:`[N,N,N,Nm,Nm,Nm]`.)doc";

static const char *mkd_doc_franky_RobotState_O_T_EE =
    R"doc(:math:`^{O}T_{EE}` Measured end effector pose in the base frame.)doc";

static const char *mkd_doc_franky_RobotState_O_T_EE_c =
    R"doc(:math:`{^OT_{EE}}_{c}` Last commanded end effector pose of motion generation in the base frame.)doc";

static const char *mkd_doc_franky_RobotState_O_T_EE_d =
    R"doc(:math:`{^OT_{EE}}_{d}` Last desired end effector pose of motion generation in the base frame.)doc";

static const char *mkd_doc_franky_RobotState_O_dP_EE_c =
    R"doc(:math:`{^OdP_{EE}}_{c}` Last commanded end effector twist in the base frame.)doc";

static const char *mkd_doc_franky_RobotState_O_dP_EE_d =
    R"doc(:math:`{^OdP_{EE}}_{d}` Desired end effector twist in the base frame.
Unit:
:math:`[\frac{m}{s},\frac{m}{s},\frac{m}{s},\frac{rad}{s},\frac{rad}{s},\frac{rad}{s}]`.)doc";

static const char *mkd_doc_franky_RobotState_O_dP_EE_est =
    R"doc(Estimated end-effector twist (linear and angular velocity) expressed in the base frame. Computed by
franky; not provided by franka firmware. Unit:
:math:`[\frac{m}{s},\frac{m}{s},\frac{m}{s},\frac{rad}{s},\frac{rad}{s},\frac{rad}{s}]`.)doc";

static const char *mkd_doc_franky_RobotState_O_ddP_EE_c =
    R"doc(:math:`{^OddP_{EE}}_{c}` Last commanded end effector acceleration in the base frame.
Unit:
:math:`[\frac{m}{s^2},\frac{m}{s^2},\frac{m}{s^2},\frac{rad}{s^2},\frac{rad}{s^2},\frac{rad}{s^2}]`.)doc";

static const char *mkd_doc_franky_RobotState_O_ddP_EE_est =
    R"doc(Estimated end-effector spatial acceleration (linear and angular acceleration) expressed in the base
frame. Computed by franky; not provided by franka firmware. Unit:
:math:`[\frac{m}{s^2},\frac{m}{s^2},\frac{m}{s^2},\frac{rad}{s^2},\frac{rad}{s^2},\frac{rad}{s^2}]`.)doc";

static const char *mkd_doc_franky_RobotState_O_ddP_O =
    R"doc(:math:`{^OddP}_O` Linear component of the acceleration of the robot's base, expressed in frame
parallel to the the base frame, i.e. the base's translational acceleration. If the base is resting
this shows the direction of the gravity vector. It is hardcoded for now to `{0, 0, -9.81}`.)doc";

static const char *mkd_doc_franky_RobotState_cartesian_collision =
    R"doc(Indicates which contact level is activated in which Cartesian dimension :math:`(x,y,z,R,P,Y)`. After
contact disappears, the value stays the same until a reset command is sent.

See also:
    Robot::setCollisionBehavior for setting sensitivity values.

See also:
    Robot::automaticErrorRecovery for performing a reset after a collision.)doc";

static const char *mkd_doc_franky_RobotState_cartesian_contact =
    R"doc(Indicates which contact level is activated in which Cartesian dimension :math:`(x,y,z,R,P,Y)`. After
contact disappears, the value turns to zero.

See also:
    Robot::setCollisionBehavior for setting sensitivity values.)doc";

static const char *mkd_doc_franky_RobotState_control_command_success_rate =
    R"doc(Percentage of the last 100 control commands that were successfully received by the robot.

Shows a value of zero if no control or motion generator loop is currently running.

Range: :math:`[0, 1]`.)doc";

static const char *mkd_doc_franky_RobotState_current_errors = R"doc(Current error state.)doc";

static const char *mkd_doc_franky_RobotState_ddelbow_c =
    R"doc(Commanded elbow acceleration of the 3rd joint in :math:`\frac{rad}{s^2}`)doc";

static const char *mkd_doc_franky_RobotState_ddelbow_est =
    R"doc(Estimated elbow acceleration (acceleration of the third joint), computed by franky. Not provided by
franka firmware. Unit: :math:`[\frac{rad}{s^2}]`.)doc";

static const char *mkd_doc_franky_RobotState_ddq_d =
    R"doc(:math:`\ddot{q}_d` Desired joint acceleration. Unit: :math:`[\frac{rad}{s^2}]`)doc";

static const char *mkd_doc_franky_RobotState_ddq_est =
    R"doc(Estimated joint acceleration computed by franky. This value does not come from the franka firmware.
Unit: :math:`[\frac{rad}{s^2}]`.)doc";

static const char *mkd_doc_franky_RobotState_delbow_c =
    R"doc(Commanded velocity of the 3rd joint in :math:`\frac{rad}{s}`)doc";

static const char *mkd_doc_franky_RobotState_delbow_est =
    R"doc(Estimated elbow velocity (velocity of the third joint), computed by franky. Not provided by franka
firmware.
Unit: :math:`[\frac{rad}{s}]`.)doc";

static const char *mkd_doc_franky_RobotState_dq =
    R"doc(:math:`\dot{q}` Measured joint velocity. Unit: :math:`[\frac{rad}{s}]`)doc";

static const char *mkd_doc_franky_RobotState_dq_d =
    R"doc(:math:`\dot{q}_d` Desired joint velocity. Unit: :math:`[\frac{rad}{s}]`)doc";

static const char *mkd_doc_franky_RobotState_dq_est =
    R"doc(Estimated joint velocity computed by franky. This value does not come from the franka firmware.
Unit: :math:`[\frac{rad}{s}]`.)doc";

static const char *mkd_doc_franky_RobotState_dtau_J =
    R"doc(:math:`\dot{\tau_{J}}` Derivative of measured link-side joint torque sensor signals. Unit:
:math:`[\frac{Nm}{s}]`)doc";

static const char *mkd_doc_franky_RobotState_dtheta =
    R"doc(:math:`\dot{\theta}` Motor velocity. Unit: :math:`[\frac{rad}{s}]`)doc";

static const char *mkd_doc_franky_RobotState_elbow = R"doc(Elbow configuration.)doc";

static const char *mkd_doc_franky_RobotState_elbow_c = R"doc(Commanded elbow configuration.)doc";

static const char *mkd_doc_franky_RobotState_elbow_d = R"doc(Desired elbow configuration.)doc";

static const char *mkd_doc_franky_RobotState_from_franka = R"doc()doc";

static const char *mkd_doc_franky_RobotState_from_franka_2 = R"doc()doc";

static const char *mkd_doc_franky_RobotState_joint_collision =
    R"doc(Indicates which contact level is activated in which joint. After contact disappears, the value stays
the same until a reset command is sent.

See also:
    Robot::setCollisionBehavior for setting sensitivity values.

See also:
    Robot::automaticErrorRecovery for performing a reset after a collision.)doc";

static const char *mkd_doc_franky_RobotState_joint_contact =
    R"doc(Indicates which contact level is activated in which joint. After contact disappears, value turns to
zero.

See also:
    Robot::setCollisionBehavior for setting sensitivity values.)doc";

static const char *mkd_doc_franky_RobotState_last_motion_errors =
    R"doc(Contains the errors that aborted the previous motion.)doc";

static const char *mkd_doc_franky_RobotState_m_ee = R"doc(:math:`m_{EE}` Configured mass of the end effector.)doc";

static const char *mkd_doc_franky_RobotState_m_load = R"doc(:math:`m_{load}` Configured mass of the external load.)doc";

static const char *mkd_doc_franky_RobotState_m_total =
    R"doc(:math:`m_{total}` Sum of the mass of the end effector and the external load.)doc";

static const char *mkd_doc_franky_RobotState_q = R"doc(:math:`q` Measured joint position. Unit: :math:`[rad]`)doc";

static const char *mkd_doc_franky_RobotState_q_d = R"doc(:math:`q_d` Desired joint position. Unit: :math:`[rad]`)doc";

static const char *mkd_doc_franky_RobotState_q_est =
    R"doc(Estimated joint position computed by franky. This value does not come from the franka firmware.
Unit: :math:`[rad]`.)doc";

static const char *mkd_doc_franky_RobotState_robot_mode = R"doc(Current robot mode.)doc";

static const char *mkd_doc_franky_RobotState_tau_J =
    R"doc(:math:`\tau_{J}` Measured link-side joint torque sensor signals. Unit: :math:`[Nm]`)doc";

static const char *mkd_doc_franky_RobotState_tau_J_d =
    R"doc(:math:`{\tau_J}_d` Desired link-side joint torque sensor signals without gravity. Unit: :math:`[Nm]`)doc";

static const char *mkd_doc_franky_RobotState_tau_ext_hat_filtered =
    R"doc(:math:`\hat{\tau}_{\text{ext}}` Low-pass filtered torques generated by external forces on the
joints. It does not include configured end-effector and load nor the mass and dynamics of the robot.
tau_ext_hat_filtered is the error between tau_J and the expected torques given by the robot model.
Unit: :math:`[Nm]`.)doc";

static const char *mkd_doc_franky_RobotState_theta = R"doc(:math:`\theta` Motor position. Unit: :math:`[rad]`)doc";

static const char *mkd_doc_franky_RobotState_time =
    R"doc(Strictly monotonically increasing timestamp since robot start.

Inside of control loops, the time_step parameter of the control loop callbacks can be used instead.)doc";

static const char *mkd_doc_franky_RobotVelocity =
    R"doc(Cartesian velocity of a robot.

This class encapsulates the cartesian velocity of a robot, which comprises the end effector twist
and the elbow velocity.)doc";

static const char *mkd_doc_franky_RobotVelocity_RobotVelocity = R"doc()doc";

static const char *mkd_doc_franky_RobotVelocity_RobotVelocity_2 = R"doc()doc";

static const char *mkd_doc_franky_RobotVelocity_RobotVelocity_3 =
    R"doc(Args:
    end_effector_twist: The twist of the end effector.
    elbow_velocity: The velocity of the elbow (optional).

)doc";

static const char *mkd_doc_franky_RobotVelocity_RobotVelocity_4 =
    R"doc(Args:
    vector_repr: The vector representation of the velocity.
    ignore_elbow: Whether to ignore the elbow velocity. Default is false.

)doc";

static const char *mkd_doc_franky_RobotVelocity_RobotVelocity_5 =
    R"doc(Args:
    vector_repr: The vector representation of the velocity.
    elbow_velocity: The velocity of the elbow (optional).

)doc";

static const char *mkd_doc_franky_RobotVelocity_RobotVelocity_6 =
    R"doc(Args:
    franka_velocity: The franka velocity.

)doc";

static const char *mkd_doc_franky_RobotVelocity_as_franka_velocity =
    R"doc(Get the franka velocity.

Args:
    elbow_state: The elbow state to use. Note, that franka::CartesianVelocities contains the elbow
                 state and not the elbow velocity, contrary to RobotVelocity.
    default_elbow_flip_direction: The default flip direction of the elbow if it is not set.

Returns:
    The franka velocity.

)doc";

static const char *mkd_doc_franky_RobotVelocity_changeEndEffectorFrame =
    R"doc(Change the end-effector frame by adding the given offset to the current end-effector frame. Note
that the offset must be given in world coordinates.

Args:
    offset_world_frame: The offset to add to the current end-effector frame.

Returns:
    The velocity of the new end-effector frame.

)doc";

static const char *mkd_doc_franky_RobotVelocity_elbow_velocity =
    R"doc(Get the elbow velocity.

Returns:
    The elbow velocity.

)doc";

static const char *mkd_doc_franky_RobotVelocity_elbow_velocity_2 = R"doc()doc";

static const char *mkd_doc_franky_RobotVelocity_end_effector_twist =
    R"doc(Get the end effector twist.

Returns:
    The end effector twist.

)doc";

static const char *mkd_doc_franky_RobotVelocity_end_effector_twist_2 = R"doc()doc";

static const char *mkd_doc_franky_RobotVelocity_transform =
    R"doc(Transform the frame of the velocity by applying the given affine transform.

Args:
    affine: The affine to apply.

Returns:
    The velocity after the transformation.

)doc";

static const char *mkd_doc_franky_RobotVelocity_transform_2 =
    R"doc(Transform the frame of the velocity by applying the given rotation.

Args:
    rotation: The rotation to apply.

Returns:
    The velocity after the transformation.

)doc";

static const char *mkd_doc_franky_RobotVelocity_vector_repr =
    R"doc(Get the vector representation of the velocity. It consists of the linear and angular velocity of the
end effector and the joint velocity of the elbow.

Returns:
    The vector representation of the velocity.

)doc";

static const char *mkd_doc_franky_RobotVelocity_withElbowVelocity =
    R"doc(Get the velocity with a new elbow velocity.

Args:
    elbow_velocity: The new elbow velocity.

Returns:
    The velocity with the new elbow velocity.

)doc";

static const char *mkd_doc_franky_Robot_Params = R"doc(Global parameters for the robot.)doc";

static const char *mkd_doc_franky_Robot_Params_controller_mode =
    R"doc(Default controller mode for the robot.

See libfranka documentation for details.)doc";

static const char *mkd_doc_franky_Robot_Params_default_force_threshold =
    R"doc(Default force threshold for collision behavior.)doc";

static const char *mkd_doc_franky_Robot_Params_default_torque_threshold =
    R"doc(Default torque threshold for collision behavior.)doc";

static const char *mkd_doc_franky_Robot_Params_kalman_control_adaptation_rate =
    R"doc(Kalman parameter: rate of adaptation of the robot state to the desired robot state.)doc";

static const char *mkd_doc_franky_Robot_Params_kalman_control_process_var =
    R"doc(Kalman parameter: process noise variance of the control signal.)doc";

static const char *mkd_doc_franky_Robot_Params_kalman_ddq_d_obs_var =
    R"doc(Kalman parameter: observation noise variance of desired joint accelerations.)doc";

static const char *mkd_doc_franky_Robot_Params_kalman_ddq_process_var =
    R"doc(Kalman parameter: process noise variance of the acceleration.)doc";

static const char *mkd_doc_franky_Robot_Params_kalman_dq_d_obs_var =
    R"doc(Kalman parameter: observation noise variance of desired joint velocities.)doc";

static const char *mkd_doc_franky_Robot_Params_kalman_dq_obs_var =
    R"doc(Kalman parameter: observation noise variance of measured joint velocities.)doc";

static const char *mkd_doc_franky_Robot_Params_kalman_dq_process_var =
    R"doc(Kalman parameter: process noise variance of the velocity.)doc";

static const char *mkd_doc_franky_Robot_Params_kalman_q_d_obs_var =
    R"doc(Kalman parameter: observation noise variance of desired joint positions.)doc";

static const char *mkd_doc_franky_Robot_Params_kalman_q_obs_var =
    R"doc(Kalman parameter: observation noise variance of measured joint positions.)doc";

static const char *mkd_doc_franky_Robot_Params_kalman_q_process_var =
    R"doc(Kalman parameter: process noise variance of the position.)doc";

static const char *mkd_doc_franky_Robot_Params_realtime_config =
    R"doc(Default realtime configuration for the robot.

See libfranka documentation for details.)doc";

static const char *mkd_doc_franky_Robot_Params_relative_dynamics_factor =
    R"doc(Relative dynamics factor for the robot.

The maximum velocity, acceleration and jerk of the robot are scaled by the factors specified here.)doc";

static const char *mkd_doc_franky_Robot_Robot =
    R"doc(Args:
    fci_hostname: The hostname or IP address of the robot.

)doc";

static const char *mkd_doc_franky_Robot_Robot_2 =
    R"doc(Args:
    fci_hostname: The hostname or IP address of the robot.
    params: The parameters for the robot.

)doc";

static const char *mkd_doc_franky_Robot_control_exception = R"doc()doc";

static const char *mkd_doc_franky_Robot_control_finished_condition = R"doc()doc";

static const char *mkd_doc_franky_Robot_control_mutex = R"doc()doc";

static const char *mkd_doc_franky_Robot_control_thread = R"doc()doc";

static const char *mkd_doc_franky_Robot_currentCartesianState =
    R"doc(Returns the current cartesian state of the robot.

Returns:
    The current cartesian state of the robot.

)doc";

static const char *mkd_doc_franky_Robot_currentCartesianVelocity =
    R"doc(Returns the current cartesian velocity of the robot.

Returns:
    The current cartesian velocity of the robot.

)doc";

static const char *mkd_doc_franky_Robot_currentJointPositions =
    R"doc(Returns the current joint positions of the robot.

Returns:
    The current joint positions of the robot.

)doc";

static const char *mkd_doc_franky_Robot_currentJointState =
    R"doc(Returns the current joint state of the robot.

Returns:
    The current joint state of the robot.

)doc";

static const char *mkd_doc_franky_Robot_currentJointVelocities =
    R"doc(Returns the current joint velocities of the robot.

Returns:
    The current joint velocities of the robot.

)doc";

static const char *mkd_doc_franky_Robot_currentPose =
    R"doc(Returns the current pose of the robot.

Returns:
    The current pose of the robot.

)doc";

static const char *mkd_doc_franky_Robot_current_control_signal_type =
    R"doc(The type of the current control signal.)doc";

static const char *mkd_doc_franky_Robot_elbow_acceleration_limit = R"doc(Elbow acceleration limit [rad/s²].)doc";

static const char *mkd_doc_franky_Robot_elbow_jerk_limit = R"doc(Elbow jerk limit [rad/s³].)doc";

static const char *mkd_doc_franky_Robot_elbow_velocity_limit = R"doc(Elbow velocity limit [rad/s].)doc";

static const char *mkd_doc_franky_Robot_fci_hostname = R"doc(The hostname of the robot.)doc";

static const char *mkd_doc_franky_Robot_fci_hostname_2 = R"doc(The robot's hostname / IP address)doc";

static const char *mkd_doc_franky_Robot_hasErrors =
    R"doc(Returns whether the robot has errors.

Returns:
    Whether the robot has errors.

)doc";

static const char *mkd_doc_franky_Robot_is_in_control =
    R"doc(Whether the robot is currently in control, i.e. a motion is being executed.)doc";

static const char *mkd_doc_franky_Robot_is_in_control_unsafe = R"doc()doc";

static const char *mkd_doc_franky_Robot_joinMotion =
    R"doc(Wait for the current motion to finish. Throw any exceptions that occurred during the motion.)doc";

static const char *mkd_doc_franky_Robot_joinMotion_2 =
    R"doc(Wait for the current motion to finish with a timeout. Throw any exceptions that occurred during the
motion.

After the timeout has expired, the function will return false.

Args:
    timeout: The timeout to wait for the motion to finish.

Returns:
    Whether the motion finished before the timeout expired.

)doc";

static const char *mkd_doc_franky_Robot_joinMotionUnsafe = R"doc()doc";

static const char *mkd_doc_franky_Robot_joint_acceleration_limit = R"doc(Joint acceleration limit [rad/s²].)doc";

static const char *mkd_doc_franky_Robot_joint_jerk_limit = R"doc(Joint jerk limit [rad/s³].)doc";

static const char *mkd_doc_franky_Robot_joint_velocity_limit = R"doc(Joint velocity limit [rad/s].)doc";

static const char *mkd_doc_franky_Robot_model =
    R"doc(The model of the robot.

The model is loaded in the constructor, so calling this function does not incur any overhead.

)doc";

static const char *mkd_doc_franky_Robot_model_2 = R"doc()doc";

static const char *mkd_doc_franky_Robot_model_urdf = R"doc(The URDF of the robot model as string.)doc";

static const char *mkd_doc_franky_Robot_model_urdf_2 = R"doc()doc";

static const char *mkd_doc_franky_Robot_motion_generator = R"doc()doc";

static const char *mkd_doc_franky_Robot_motion_generator_running = R"doc()doc";

static const char *mkd_doc_franky_Robot_move =
    R"doc(Execute the given motion

Args:
    motion: The motion to execute.
    async: Whether to execute the motion asynchronously.
    limit_rate: True if rate limiting should be activated. False by default. This could distort your
                motion!
    cutoff_frequency: Cutoff frequency for a first order low-pass filter applied on the user
                      commanded signal. Set to franka::kMaxCutoffFrequency to disable.

)doc";

static const char *mkd_doc_franky_Robot_move_2 =
    R"doc(Execute the given motion

Args:
    motion: The motion to execute.
    async: Whether to execute the motion asynchronously.
    limit_rate: True if rate limiting should be activated. False by default. This could distort your
                motion!
    cutoff_frequency: Cutoff frequency for a first order low-pass filter applied on the user
                      commanded signal. Set to franka::kMaxCutoffFrequency to disable.

)doc";

static const char *mkd_doc_franky_Robot_move_3 =
    R"doc(Execute the given motion

Args:
    motion: The motion to execute.
    async: Whether to execute the motion asynchronously.
    limit_rate: True if rate limiting should be activated. False by default. This could distort your
                motion!
    cutoff_frequency: Cutoff frequency for a first order low-pass filter applied on the user
                      commanded signal. Set to franka::kMaxCutoffFrequency to disable.

)doc";

static const char *mkd_doc_franky_Robot_move_4 =
    R"doc(Execute the given motion

Args:
    motion: The motion to execute.
    async: Whether to execute the motion asynchronously.
    limit_rate: True if rate limiting should be activated. False by default. This could distort your
                motion!
    cutoff_frequency: Cutoff frequency for a first order low-pass filter applied on the user
                      commanded signal. Set to franka::kMaxCutoffFrequency to disable.

)doc";

static const char *mkd_doc_franky_Robot_move_5 =
    R"doc(Execute the given motion

Args:
    motion: The motion to execute.
    async: Whether to execute the motion asynchronously.
    limit_rate: True if rate limiting should be activated. False by default. This could distort your
                motion!
    cutoff_frequency: Cutoff frequency for a first order low-pass filter applied on the user
                      commanded signal. Set to franka::kMaxCutoffFrequency to disable.

)doc";

static const char *mkd_doc_franky_Robot_moveInternal = R"doc()doc";

static const char *mkd_doc_franky_Robot_params = R"doc()doc";

static const char *mkd_doc_franky_Robot_pollMotion =
    R"doc(Check whether the robot is still in motion. This function is non-blocking and returns immediately.
Throw any exceptions that occurred during the motion.

Returns:
    Whether the robot is still in motion.

)doc";

static const char *mkd_doc_franky_Robot_recoverFromErrors =
    R"doc(Calls the automatic error recovery of the robot and returns whether the recovery was successful.

Returns:
    Whether the recovery was successful.

)doc";

static const char *mkd_doc_franky_Robot_relative_dynamics_factor =
    R"doc(Returns the current global relative dynamics factor of the robot.

Returns:
    The current relative dynamics factor of the robot.

)doc";

static const char *mkd_doc_franky_Robot_relative_dynamics_factor_handle = R"doc()doc";

static const char *mkd_doc_franky_Robot_relative_dynamics_factor_rt =
    R"doc(Returns the current global relative dynamics factor of the robot (Real-Time safe).

Returns:
    The current relative dynamics factor of the robot.

)doc";

static const char *mkd_doc_franky_Robot_rotation_acceleration_limit =
    R"doc(Rotational acceleration limit [rad/s²].)doc";

static const char *mkd_doc_franky_Robot_rotation_jerk_limit = R"doc(Rotational jerk limit [rad/s³].)doc";

static const char *mkd_doc_franky_Robot_rotation_velocity_limit = R"doc(Rotational velocity limit [rad/s].)doc";

static const char *mkd_doc_franky_Robot_setCollisionBehavior =
    R"doc(Set the collision behavior of the robot.

Args:
    torque_threshold: The torque threshold for the collision behavior in Nm.
    force_threshold: The force threshold for the collision behavior in N.

)doc";

static const char *mkd_doc_franky_Robot_setCollisionBehavior_2 =
    R"doc(Set the collision behavior of the robot.

Args:
    lower_torque_threshold: The lower torque threshold for the collision behavior in Nm.
    upper_torque_threshold: The upper torque threshold for the collision behavior in Nm.
    lower_force_threshold: The lower force threshold for the collision behavior in N.
    upper_force_threshold: The upper force threshold for the collision behavior in N.

)doc";

static const char *mkd_doc_franky_Robot_setCollisionBehavior_3 =
    R"doc(Set the collision behavior of the robot.

Args:
    lower_torque_threshold_acceleration: The lower torque threshold for the collision behavior in Nm
                                         during acceleration.
    upper_torque_threshold_acceleration: The upper torque threshold for the collision behavior in Nm
                                         during acceleration.
    lower_torque_threshold_nominal: The lower torque threshold for the collision behavior in Nm
                                    during nominal operation.
    upper_torque_threshold_nominal: The upper torque threshold for the collision behavior in Nm
                                    during nominal operation.
    lower_force_threshold_acceleration: The lower force threshold for the collision behavior in N
                                        during acceleration.
    upper_force_threshold_acceleration: The upper force threshold for the collision behavior in N
                                        during acceleration.
    lower_force_threshold_nominal: The lower force threshold for the collision behavior in N during
                                   nominal operation.
    upper_force_threshold_nominal: The upper force threshold for the collision behavior in N during
                                   nominal operation.

)doc";

static const char *mkd_doc_franky_Robot_setRelativeDynamicsFactor =
    R"doc(Sets the global relative dynamics factor of the robot.

Args:
    relative_dynamics_factor: The relative dynamics factor to set.

)doc";

static const char *mkd_doc_franky_Robot_state =
    R"doc(Returns the current state of the robot.

Returns:
    The current state of the robot.

)doc";

static const char *mkd_doc_franky_Robot_state_buffer = R"doc()doc";

static const char *mkd_doc_franky_Robot_state_received_at = R"doc()doc";

static const char *mkd_doc_franky_Robot_translation_acceleration_limit =
    R"doc(Translational acceleration limit [m/s²].)doc";

static const char *mkd_doc_franky_Robot_translation_jerk_limit = R"doc(Translational jerk limit [m/s³].)doc";

static const char *mkd_doc_franky_Robot_translation_velocity_limit = R"doc(Translational velocity limit [m/s].)doc";

static const char *mkd_doc_franky_SimpleTorqueMotion =
    R"doc(Direct joint torque control with a signal watchdog.

This motion applies the joint torques it is given, without any feedback law on top. Until the first
torque signal arrives, it applies SimpleTorqueParams::initial_torque. New torques are published with
setTorque from a user thread, analogous to how the references of the impedance motions are updated,
and are picked up by the control loop in the next cycle.

As the robot is not commanded by any controller in between torque signals, the motion terminates
with a TorqueSignalTimeoutException if no new torque arrives within
SimpleTorqueParams::signal_timeout (50ms by default). Hence, the torque has to be updated regularly,
even if it does not change.

Note that the commanded torques are still subject to the torque rate limit
(TorqueSafetyParams::max_delta_tau), so large steps are ramped in over multiple cycles.)doc";

static const char *mkd_doc_franky_SimpleTorqueMotion_SimpleTorqueMotion = R"doc()doc";

static const char *mkd_doc_franky_SimpleTorqueMotion_SimpleTorqueMotion_2 =
    R"doc(Args:
    initial_torque: The torque applied until the first torque signal arrives [Nm].
    signal_timeout: Maximum duration [s] without a new torque signal, or nullopt to disable the
                    watchdog.

)doc";

static const char *mkd_doc_franky_SimpleTorqueMotion_current_torque = R"doc()doc";

static const char *mkd_doc_franky_SimpleTorqueMotion_getTorque =
    R"doc(Get a copy of the last published torque [Nm], or the initial torque if no torque has been published
yet.

)doc";

static const char *mkd_doc_franky_SimpleTorqueMotion_initImpl = R"doc()doc";

static const char *mkd_doc_franky_SimpleTorqueMotion_last_signal_time = R"doc()doc";

static const char *mkd_doc_franky_SimpleTorqueMotion_nextCommandImpl = R"doc()doc";

static const char *mkd_doc_franky_SimpleTorqueMotion_params = R"doc(The parameters of the motion.)doc";

static const char *mkd_doc_franky_SimpleTorqueMotion_params_2 = R"doc()doc";

static const char *mkd_doc_franky_SimpleTorqueMotion_setTorque =
    R"doc(Set the torque applied by the motion [Nm].

The torque is validated and picked up by the control loop in the next cycle. It also resets the
watchdog, so this function has to be called at least every SimpleTorqueParams::signal_timeout
seconds while the motion is running.

Args:
    torque: The new joint torques [Nm].

)doc";

static const char *mkd_doc_franky_SimpleTorqueMotion_torque_handle = R"doc()doc";

static const char *mkd_doc_franky_SimpleTorqueParams = R"doc(Parameters for SimpleTorqueMotion.)doc";

static const char *mkd_doc_franky_SimpleTorqueParams_compensate_coriolis =
    R"doc(Compensate Coriolis forces using the robot model.)doc";

static const char *mkd_doc_franky_SimpleTorqueParams_friction = R"doc(Joint friction compensation settings.)doc";

static const char *mkd_doc_franky_SimpleTorqueParams_initial_torque =
    R"doc(Torque applied until the first torque signal arrives [Nm].)doc";

static const char *mkd_doc_franky_SimpleTorqueParams_safety =
    R"doc(Shared torque safety limits and soft joint-limit repulsion settings.)doc";

static const char *mkd_doc_franky_SimpleTorqueParams_signal_timeout =
    R"doc(Maximum duration [s] the motion tolerates without receiving a new torque signal before it terminates
with a TorqueSignalTimeoutException. The watchdog is armed when the motion starts, so the initial
torque is only held for this duration as well. Set to nullopt to disable the watchdog and hold the
last commanded torque indefinitely.)doc";

static const char *mkd_doc_franky_SimpleTorqueParams_validate =
    R"doc(Throw std::invalid_argument if any parameter is out of range.)doc";

static const char *mkd_doc_franky_StopMotion = R"doc()doc";

static const char *mkd_doc_franky_StopMotion_2 = R"doc(Stop motion for joint position control mode.)doc";

static const char *mkd_doc_franky_StopMotion_3 = R"doc(Stop motion for joint velocity control mode.)doc";

static const char *mkd_doc_franky_StopMotion_4 = R"doc(Stop motion for cartesian pose control mode.)doc";

static const char *mkd_doc_franky_StopMotion_5 = R"doc(Stop motion for cartesian velocity control mode.)doc";

static const char *mkd_doc_franky_StopMotion_6 =
    R"doc(Graceful stop for torque-control (impedance) motions.

Torque motions never signal MotionFinished on their own, and franka::Robot::stop() preempts the
control loop with a franka::ControlException rather than ramping down. This motion ramps the last
commanded torque into a damping-only law and then returns franka::MotionFinished.)doc";

static const char *mkd_doc_franky_StopMotion_StopMotion =
    R"doc(Args:
    relative_dynamics_factor: Relative dynamics factor for this stop motion.

)doc";

static const char *mkd_doc_franky_StopMotion_StopMotion_2 =
    R"doc(Args:
    relative_dynamics_factor: Relative dynamics factor for this stop motion.

)doc";

static const char *mkd_doc_franky_StopMotion_StopMotion_3 =
    R"doc(Args:
    relative_dynamics_factor: Relative dynamics factor for this stop motion.

)doc";

static const char *mkd_doc_franky_StopMotion_StopMotion_4 =
    R"doc(Args:
    relative_dynamics_factor: Relative dynamics factor for this stop motion.

)doc";

static const char *mkd_doc_franky_StopMotion_StopMotion_5 = R"doc()doc";

static const char *mkd_doc_franky_StopMotion_initImpl = R"doc()doc";

static const char *mkd_doc_franky_StopMotion_nextCommandImpl = R"doc()doc";

static const char *mkd_doc_franky_StopMotion_params = R"doc()doc";

static const char *mkd_doc_franky_StopMotion_tau_start = R"doc()doc";

static const char *mkd_doc_franky_TorqueSafetyParams =
    R"doc(Shared torque safety limits and soft joint-limit repulsion settings for torque-control motions.)doc";

static const char *mkd_doc_franky_TorqueSafetyParams_joint_limit_activation_distance =
    R"doc(Activation distance from a limit in [rad].)doc";

static const char *mkd_doc_franky_TorqueSafetyParams_joint_limit_damping =
    R"doc(Additional damping when moving into a limit in [Nms/rad].)doc";

static const char *mkd_doc_franky_TorqueSafetyParams_joint_limit_max_torque =
    R"doc(Absolute torque clamp for the repulsion term in [Nm].)doc";

static const char *mkd_doc_franky_TorqueSafetyParams_joint_limit_stiffness = R"doc(Base repulsion gain in [Nm].)doc";

static const char *mkd_doc_franky_TorqueSafetyParams_lower_joint_limits =
    R"doc(Lower soft joint limits in [rad]. Joint-limit repulsion is active when both limits are set.)doc";

static const char *mkd_doc_franky_TorqueSafetyParams_max_delta_tau =
    R"doc(Maximum allowed torque step per cycle in [Nm].)doc";

static const char *mkd_doc_franky_TorqueSafetyParams_upper_joint_limits =
    R"doc(Upper soft joint limits in [rad]. Joint-limit repulsion is active when both limits are set.)doc";

static const char *mkd_doc_franky_TorqueSafetyParams_validate =
    R"doc(Throw std::invalid_argument if any safety parameter is out of range.)doc";

static const char *mkd_doc_franky_TorqueSignalTimeoutException =
    R"doc(Thrown when a SimpleTorqueMotion does not receive a new torque signal in time.)doc";

static const char *mkd_doc_franky_TorqueStopParams =
    R"doc(Parameters for the torque-control stop motion (StopMotion<franka::Torques>).)doc";

static const char *mkd_doc_franky_TorqueStopParams_compensate_coriolis = R"doc(Compensate Coriolis forces.)doc";

static const char *mkd_doc_franky_TorqueStopParams_damping =
    R"doc(Pure joint damping [Nms/rad] used to bring the arm to rest.)doc";

static const char *mkd_doc_franky_TorqueStopParams_max_delta_tau =
    R"doc(Per-tick torque-rate limit [Nm] for smoothness.)doc";

static const char *mkd_doc_franky_TorqueStopParams_max_duration =
    R"doc(Maximum duration [s] before the motion finishes regardless of velocity.)doc";

static const char *mkd_doc_franky_TorqueStopParams_ramp_duration =
    R"doc(Duration [s] over which the inherited command is ramped into the damping-only law. Set to 0 to
switch immediately.)doc";

static const char *mkd_doc_franky_TorqueStopParams_velocity_epsilon =
    R"doc(Finish once every joint speed is below this [rad/s] (after the ramp completes).)doc";

static const char *mkd_doc_franky_Twist = R"doc(Twist of a frame.)doc";

static const char *mkd_doc_franky_TwistAcceleration =
    R"doc(Acceleration of a frame (2nd time derivative of a pose, consisting of a linear and an angular
acceleration).)doc";

static const char *mkd_doc_franky_TwistAcceleration_TwistAcceleration =
    R"doc(Args:
    linear_acceleration: The linear acceleration in [m/s^2].
    angular_acceleration: The angular acceleration in [rad/s^2].

)doc";

static const char *mkd_doc_franky_TwistAcceleration_angular_acceleration =
    R"doc(Get the angular acceleration.

Returns:
    The angular acceleration [rad/s^2].

)doc";

static const char *mkd_doc_franky_TwistAcceleration_angular_acceleration_2 = R"doc()doc";

static const char *mkd_doc_franky_TwistAcceleration_fromVectorRepr =
    R"doc(Args:
    vector_repr: The vector representation of the twist acceleration.

)doc";

static const char *mkd_doc_franky_TwistAcceleration_linear_acceleration =
    R"doc(Get the linear acceleration.

Returns:
    The linear acceleration [m/s^2].

)doc";

static const char *mkd_doc_franky_TwistAcceleration_linear_acceleration_2 = R"doc()doc";

static const char *mkd_doc_franky_TwistAcceleration_propagateThroughLink =
    R"doc(Propagate the twist acceleration through a link with the given translation. Hence, suppose this
twist acceleration is the twist acceleration of a frame A, then this function computes the twist
acceleration of a frame B that is rigidly attached to frame A by a link with the given translation:
B = A + T, where T is the translation.

Args:
    link_translation: The translation of the link (A to B). Must be in the same reference frame as
                      this twist acceleration.
    base_angular_velocity: Angular velocity of the base frame (A) [rad/s]. Must be in the same
                           reference frame as this twist acceleration.

Returns:
    The twist acceleration propagated through the link.

)doc";

static const char *mkd_doc_franky_TwistAcceleration_transformWith =
    R"doc(Transform the frame of the twist acceleration by applying the given affine transform.

Args:
    transformation: The transformation to apply.

Returns:
    The twist acceleration after the transformation.

)doc";

static const char *mkd_doc_franky_TwistAcceleration_transformWith_2 =
    R"doc(Transform the frame of the twist acceleration by applying the given rotation.

Args:
    rotation: The rotation to apply.

Returns:
    The twist acceleration after the transformation.

)doc";

static const char *mkd_doc_franky_TwistAcceleration_vector_repr =
    R"doc(Get the vector representation of the twist acceleration. It consists of the linear and angular
velocities.

Returns:
    The vector representation of the twist acceleration.

)doc";

static const char *mkd_doc_franky_Twist_Twist =
    R"doc(Args:
    linear_velocity: The linear velocity in [m/s].
    angular_velocity: The angular velocity in [rad/s].

)doc";

static const char *mkd_doc_franky_Twist_angular_velocity =
    R"doc(Get the angular velocity.

Returns:
    The angular velocity [rad/s].

)doc";

static const char *mkd_doc_franky_Twist_angular_velocity_2 = R"doc()doc";

static const char *mkd_doc_franky_Twist_fromVectorRepr =
    R"doc(Args:
    vector_repr: The vector representation of the twist.

)doc";

static const char *mkd_doc_franky_Twist_linear_velocity =
    R"doc(Get the linear velocity.

Returns:
    The linear velocity [m/s].

)doc";

static const char *mkd_doc_franky_Twist_linear_velocity_2 = R"doc()doc";

static const char *mkd_doc_franky_Twist_propagateThroughLink =
    R"doc(Propagate the twist through a link with the given translation.

Hence, suppose this twist is the twist of a frame A, then this function computes the twist of a
frame B that is rigidly attached to frame A by a link with the given translation: B = A + T, where T
is the translation.

Args:
    link_translation: The translation of the link. Must be in the same reference frame as this
                      twist.

Returns:
    The twist propagated through the link.

)doc";

static const char *mkd_doc_franky_Twist_transformWith =
    R"doc(Transform the frame of the twist by applying the given affine transform.

Args:
    transformation: The transformation to apply.

Returns:
    The twist after the transformation.

)doc";

static const char *mkd_doc_franky_Twist_transformWith_2 =
    R"doc(Transform the frame of the twist by applying the given rotation.

Args:
    rotation: The rotation to apply.

Returns:
    The twist after the transformation.

)doc";

static const char *mkd_doc_franky_Twist_vector_repr =
    R"doc(Get the vector representation of the twist. It consists of the linear and angular velocities.

Returns:
    The vector representation of the twist.

)doc";

static const char *mkd_doc_franky_VelocityWaypointMotion =
    R"doc(A motion following multiple velocity waypoints in a time-optimal way. Works with arbitrary initial
conditions.

Template Args:
    ControlSignalType: The type of the control signal. Either franka::Torques,
                       franka::JointVelocities, franka::CartesianVelocities, franka::JointPositions
                       or franka::CartesianPose.
    TargetType: The type of the target of the waypoints.)doc";

static const char *mkd_doc_franky_VelocityWaypointMotion_VelocityWaypointMotion =
    R"doc(Args:
    waypoints: The waypoints to follow.
    relative_dynamics_factor: The relative dynamics factor for this motion. This factor will get
                              multiplied with the robot's global dynamics factor to get the actual
                              dynamics factor for this motion.

)doc";

static const char *mkd_doc_franky_VelocityWaypointMotion_extrapolateMotion = R"doc()doc";

static const char *mkd_doc_franky_VelocityWaypointMotion_getAbsoluteInputLimits = R"doc()doc";

static const char *mkd_doc_franky_VelocityWaypointMotion_getDesiredState = R"doc()doc";

static const char *mkd_doc_franky_VelocityWaypointMotion_relative_dynamics_factor =
    R"doc(The relative dynamics factor of this motion.)doc";

static const char *mkd_doc_franky_VelocityWaypointMotion_relative_dynamics_factor_2 = R"doc()doc";

static const char *mkd_doc_franky_VelocityWaypointMotion_setInputLimits = R"doc()doc";

static const char *mkd_doc_franky_WaitFreeTripleBuffer =
    R"doc(Triple buffer connecting concurrent user threads with a single real-time thread.

The core exchange is a wait-free, Single-Producer Single-Consumer (SPSC) triple buffer: one writer
can publish data without ever blocking, and one reader can obtain the latest available data without
locking or experiencing torn reads. On top of that, a write mutex and a read mutex serialize
concurrent writers and concurrent readers, respectively, so that multiple user threads may safely
share either side.

The real-time thread must not take these mutexes, as it could then be blocked by user threads.
Instead, it must exclusively use the unsafe variants (setUnsafe(), getUnsafe(),
getLastWrittenUnsafe()), which is safe as long as it is the only thread on its side of the
buffer: all other threads on the same side must use the locking variants, and no other thread
may use the unsafe variants concurrently.)doc";

static const char *mkd_doc_franky_WaitFreeTripleBuffer_WaitFreeTripleBuffer = R"doc()doc";

static const char *mkd_doc_franky_WaitFreeTripleBuffer_WaitFreeTripleBuffer_2 = R"doc()doc";

static const char *mkd_doc_franky_WaitFreeTripleBuffer_active_read_index = R"doc()doc";

static const char *mkd_doc_franky_WaitFreeTripleBuffer_active_write_index = R"doc()doc";

static const char *mkd_doc_franky_WaitFreeTripleBuffer_buffers = R"doc()doc";

static const char *mkd_doc_franky_WaitFreeTripleBuffer_get =
    R"doc(Get the most recently published data.

Takes the read mutex to serialize concurrent readers. Must not be called from the real-time thread;
use getUnsafe() there instead.

Note: This method mutates internal read indices to securely take ownership of the newest
buffer, hence it cannot be marked const.

)doc";

static const char *mkd_doc_franky_WaitFreeTripleBuffer_getLastWritten =
    R"doc(Get the most recently written value from the writer's side.

Takes the write mutex, as it accesses writer-owned state. Unlike get(), this does not consume the
"new data" flag or interact with the reader in any way. Before the first set(), this returns a
default-constructed T. Must not be called from the real-time thread; use getLastWrittenUnsafe()
there instead.

)doc";

static const char *mkd_doc_franky_WaitFreeTripleBuffer_getLastWrittenUnsafe =
    R"doc(Get the most recently written value from the writer's side without taking the write mutex.

Must only be called if no other thread is writing concurrently. This is the variant the real-time
thread has to use, as it must not block on the write mutex.

)doc";

static const char *mkd_doc_franky_WaitFreeTripleBuffer_getUnsafe =
    R"doc(Get the most recently published data without taking the read mutex.

Must only be called if no other thread is reading concurrently. This is the variant the real-time
thread has to use, as it must not block on the read mutex.

)doc";

static const char *mkd_doc_franky_WaitFreeTripleBuffer_hasNewData =
    R"doc(Check whether the writer has published data that no reader has consumed yet.

This is a single atomic load and does not consume the "new data" flag; only get() and getUnsafe()
do. Hence, it is safe to call from any thread, including the real-time thread.

)doc";

static const char *mkd_doc_franky_WaitFreeTripleBuffer_last_written_index = R"doc()doc";

static const char *mkd_doc_franky_WaitFreeTripleBuffer_read_mutex = R"doc()doc";

static const char *mkd_doc_franky_WaitFreeTripleBuffer_set =
    R"doc(Publish new data.

Takes the write mutex to serialize concurrent writers. Must not be called from the real-time thread;
use setUnsafe() there instead.

)doc";

static const char *mkd_doc_franky_WaitFreeTripleBuffer_setUnsafe =
    R"doc(Publish new data without taking the write mutex.

Must only be called if no other thread is writing concurrently. This is the variant the real-time
thread has to use, as it must not block on the write mutex.

)doc";

static const char *mkd_doc_franky_WaitFreeTripleBuffer_shared_state = R"doc()doc";

static const char *mkd_doc_franky_WaitFreeTripleBuffer_write_mutex = R"doc()doc";

static const char *mkd_doc_franky_Waypoint =
    R"doc(A waypoint with a target and optional parameters.

Template Args:
    TargetType: The type of the target.)doc";

static const char *mkd_doc_franky_WaypointMotion =
    R"doc(A motion following multiple waypoints in a time-optimal way. Works with arbitrary initial
conditions.

Template Args:
    ControlSignalType: The type of the control signal. Either franka::Torques,
                       franka::JointVelocities, franka::CartesianVelocities, franka::JointPositions
                       or franka::CartesianPose.
    WaypointType: The type of the waypoints. Must subclass Waypoint<TargetType>.
    TargetType: The type of the target of the waypoints.)doc";

static const char *mkd_doc_franky_WaypointMotion_WaypointMotion =
    R"doc(Args:
    waypoints: The waypoints to follow.
    return_when_finished: Whether to end the motion when the last waypoint is reached or keep
                          holding the last target.

)doc";

static const char *mkd_doc_franky_WaypointMotion_checkWaypoint = R"doc()doc";

static const char *mkd_doc_franky_WaypointMotion_extrapolateMotion = R"doc()doc";

static const char *mkd_doc_franky_WaypointMotion_getAbsoluteInputLimits = R"doc()doc";

static const char *mkd_doc_franky_WaypointMotion_getControlSignal = R"doc()doc";

static const char *mkd_doc_franky_WaypointMotion_initImpl = R"doc()doc";

static const char *mkd_doc_franky_WaypointMotion_initWaypointMotion = R"doc()doc";

static const char *mkd_doc_franky_WaypointMotion_input_parameter = R"doc()doc";

static const char *mkd_doc_franky_WaypointMotion_nextCommandImpl = R"doc()doc";

static const char *mkd_doc_franky_WaypointMotion_output_parameter = R"doc()doc";

static const char *mkd_doc_franky_WaypointMotion_prev_result = R"doc()doc";

static const char *mkd_doc_franky_WaypointMotion_return_when_finished =
    R"doc(Whether the motion ends when the last waypoint is reached or keeps holding the last target.)doc";

static const char *mkd_doc_franky_WaypointMotion_return_when_finished_2 = R"doc()doc";

static const char *mkd_doc_franky_WaypointMotion_setInputLimits = R"doc()doc";

static const char *mkd_doc_franky_WaypointMotion_setNewWaypoint = R"doc()doc";

static const char *mkd_doc_franky_WaypointMotion_target_reached_time = R"doc()doc";

static const char *mkd_doc_franky_WaypointMotion_trajectory_generator = R"doc()doc";

static const char *mkd_doc_franky_WaypointMotion_waypoint_iterator = R"doc()doc";

static const char *mkd_doc_franky_WaypointMotion_waypoint_started_time = R"doc()doc";

static const char *mkd_doc_franky_WaypointMotion_waypoints = R"doc(The waypoints this motion follows.)doc";

static const char *mkd_doc_franky_WaypointMotion_waypoints_2 = R"doc()doc";

static const char *mkd_doc_franky_Waypoint_hold_target_duration =
    R"doc(For how long to hold the target of this waypoint after it has been reached.)doc";

static const char *mkd_doc_franky_Waypoint_max_total_duration =
    R"doc(The maximum time to try reaching this waypoint before moving on to the next waypoint. Default is
infinite.)doc";

static const char *mkd_doc_franky_Waypoint_minimum_time =
    R"doc(The minimum time to reach this waypoint. This value is ignored when the effective relative dynamics
factor has max_dynamics() set.)doc";

static const char *mkd_doc_franky_Waypoint_relative_dynamics_factor =
    R"doc(The relative dynamics factor for this waypoint. This factor will get multiplied with the robot's
global dynamics factor and the motion dynamics factor to get the actual dynamics factor for this
waypoint.)doc";

static const char *mkd_doc_franky_Waypoint_target = R"doc(The target of this waypoint.)doc";

static const char *mkd_doc_franky_cartesianGainBlocks = R"doc()doc";

static const char *mkd_doc_franky_checkRes =
    R"doc(Common helper function to handle error checking for system calls.

Args:
    res: Result of the call.
    msg: Message to print for error.

)doc";

static const char *mkd_doc_franky_computeFrictionCompensation = R"doc()doc";

static const char *mkd_doc_franky_computeJointLimitTorque = R"doc()doc";

static const char *mkd_doc_franky_defaultCartesianImpedanceDamping = R"doc()doc";

static const char *mkd_doc_franky_defaultCartesianImpedanceStiffness = R"doc()doc";

static const char *mkd_doc_franky_defaultJointImpedanceDamping = R"doc()doc";

static const char *mkd_doc_franky_defaultJointImpedanceDamping_2 = R"doc()doc";

static const char *mkd_doc_franky_defaultJointImpedanceStiffness = R"doc()doc";

static const char *mkd_doc_franky_ensureEigen = R"doc()doc";

static const char *mkd_doc_franky_ensureStd = R"doc()doc";

static const char *mkd_doc_franky_expand = R"doc()doc";

static const char *mkd_doc_franky_expandEigen = R"doc()doc";

static const char *mkd_doc_franky_interpolateGain = R"doc()doc";

static const char *mkd_doc_franky_interpolateGain_2 =
    R"doc(Interpolate a fixed-size Eigen value and snap it to the target once settled.

Snapping prevents exponentially filtered gains from changing by tiny amounts forever, which is
important when a derived quantity is cached using exact equality.

)doc";

static const char *mkd_doc_franky_measure_pow = R"doc()doc";

static const char *mkd_doc_franky_operator_add = R"doc()doc";

static const char *mkd_doc_franky_operator_div = R"doc()doc";

static const char *mkd_doc_franky_operator_eq = R"doc()doc";

static const char *mkd_doc_franky_operator_eq_2 = R"doc()doc";

static const char *mkd_doc_franky_operator_ge = R"doc()doc";

static const char *mkd_doc_franky_operator_gt = R"doc()doc";

static const char *mkd_doc_franky_operator_land = R"doc()doc";

static const char *mkd_doc_franky_operator_le = R"doc()doc";

static const char *mkd_doc_franky_operator_lnot = R"doc()doc";

static const char *mkd_doc_franky_operator_lor = R"doc()doc";

static const char *mkd_doc_franky_operator_lshift = R"doc()doc";

static const char *mkd_doc_franky_operator_lshift_2 = R"doc()doc";

static const char *mkd_doc_franky_operator_lshift_3 = R"doc()doc";

static const char *mkd_doc_franky_operator_lshift_4 = R"doc()doc";

static const char *mkd_doc_franky_operator_lshift_5 = R"doc()doc";

static const char *mkd_doc_franky_operator_lshift_6 = R"doc()doc";

static const char *mkd_doc_franky_operator_lshift_7 = R"doc()doc";

static const char *mkd_doc_franky_operator_lshift_8 = R"doc()doc";

static const char *mkd_doc_franky_operator_lshift_9 = R"doc()doc";

static const char *mkd_doc_franky_operator_lt = R"doc()doc";

static const char *mkd_doc_franky_operator_mul = R"doc()doc";

static const char *mkd_doc_franky_operator_mul_2 = R"doc()doc";

static const char *mkd_doc_franky_operator_mul_3 = R"doc()doc";

static const char *mkd_doc_franky_operator_mul_4 = R"doc()doc";

static const char *mkd_doc_franky_operator_mul_5 = R"doc()doc";

static const char *mkd_doc_franky_operator_mul_6 = R"doc()doc";

static const char *mkd_doc_franky_operator_mul_7 = R"doc()doc";

static const char *mkd_doc_franky_operator_mul_8 = R"doc()doc";

static const char *mkd_doc_franky_operator_mul_9 = R"doc()doc";

static const char *mkd_doc_franky_operator_mul_10 = R"doc()doc";

static const char *mkd_doc_franky_operator_ne = R"doc()doc";

static const char *mkd_doc_franky_operator_ne_2 = R"doc()doc";

static const char *mkd_doc_franky_operator_sub = R"doc()doc";

static const char *mkd_doc_franky_operator_sub_2 = R"doc()doc";

static const char *mkd_doc_franky_patchMutexRT =
    R"doc(Patch std::mutex to allow for priority inheritance.

Priority inheritance ensures that the thread currently holding this mutex receives the same priority
as the highest priority thread waiting for it.

Args:
    mutex: Mutex to patch.

)doc";

static const char *mkd_doc_franky_saturateTorqueRate = R"doc()doc";

static const char *mkd_doc_franky_stdToAffine = R"doc()doc";

static const char *mkd_doc_franky_toEigen = R"doc()doc";

static const char *mkd_doc_franky_toEigenD = R"doc()doc";

static const char *mkd_doc_franky_toEigenMatD = R"doc()doc";

static const char *mkd_doc_franky_toStd = R"doc()doc";

static const char *mkd_doc_franky_toStdD = R"doc()doc";

static const char *mkd_doc_franky_toStdDMatD = R"doc()doc";

static const char *mkd_doc_franky_validateFinite =
    R"doc(Throw std::invalid_argument if any element of values is non-finite.)doc";

static const char *mkd_doc_franky_validateNonNegativeFinite =
    R"doc(Throw std::invalid_argument if value is negative or non-finite.)doc";

static const char *mkd_doc_franky_validateNonNegativeFinite_2 =
    R"doc(Throw std::invalid_argument if any element of values is negative or non-finite.)doc";

static const char *mkd_doc_franky_validatePositiveSemidefinite = R"doc()doc";

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
