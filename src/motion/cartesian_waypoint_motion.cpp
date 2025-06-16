#include "franky/motion/cartesian_waypoint_motion.hpp"

#include <ruckig/ruckig.hpp>
#include <utility>

#include "franky/cartesian_state.hpp"
#include "franky/robot.hpp"
#include "franky/util.hpp"

namespace franky {

CartesianWaypointMotion::CartesianWaypointMotion(
    const std::vector<PositionWaypoint<CartesianState>> &waypoints,
    const RelativeDynamicsFactor &relative_dynamics_factor, bool return_when_finished, Affine ee_frame)
    : PositionWaypointMotion(waypoints, relative_dynamics_factor, return_when_finished),
      ee_frame_(std::move(ee_frame)) {}

void CartesianWaypointMotion::initWaypointMotion(
    const RobotState &robot_state, const std::optional<franka::CartesianPose> &previous_command,
    ruckig::InputParameter<7> &input_parameter) {
  RobotPose robot_pose;
  if (previous_command.has_value()) {
    robot_pose = RobotPose(previous_command.value());
  } else {
    robot_pose = RobotPose(robot_state.O_T_EE_c, robot_state.elbow_c);
  }
  ref_frame_ = Affine::Identity();

  const RobotVelocity initial_velocity(robot_state.O_dP_EE_c, robot_state.delbow_c);

  Vector7d initial_acceleration_with_elbow =
      (Vector7d() << robot_state.O_ddP_EE_c.vector_repr(), robot_state.ddelbow_c).finished();

  target_state_ = robot_pose;

  input_parameter.current_position = toStdD<7>(robot_pose.vector_repr());
  input_parameter.current_velocity = toStdD<7>(initial_velocity.vector_repr());
  input_parameter.current_acceleration = toStdD<7>(initial_acceleration_with_elbow);
}

franka::CartesianPose CartesianWaypointMotion::getControlSignal(
    const RobotState &robot_state, const franka::Duration &time_step,
    const std::optional<franka::CartesianPose> &previous_command, const ruckig::InputParameter<7> &input_parameter) {
  auto has_elbow = input_parameter.enabled[6];
  auto current_elbow_flip = robot_state.elbow.joint_4_flip().value();
  if (previous_command.has_value() && previous_command->hasElbow()) {
    current_elbow_flip = static_cast<FlipDirection>(previous_command->elbow[1]);
  }
  RobotPose target_pose(toEigenD<7>(input_parameter.current_position), !has_elbow);
  return (ref_frame_ * target_pose).as_franka_pose(current_elbow_flip);
}

void CartesianWaypointMotion::setNewWaypoint(
    const RobotState &robot_state, const std::optional<franka::CartesianPose> &previous_command,
    const PositionWaypoint<CartesianState> &new_waypoint, ruckig::InputParameter<7> &input_parameter) {
  auto waypoint_has_elbow = input_parameter.enabled[6];

  // We first convert the current state into the frame of the current pose
  RobotPose current_pose_old_ref_frame(toEigenD<7>(input_parameter.current_position), !waypoint_has_elbow);
  Affine new_ref_to_old_ref = current_pose_old_ref_frame.end_effector_pose();
  ref_frame_ = ref_frame_ * new_ref_to_old_ref;
  auto rot = new_ref_to_old_ref.inverse().rotation();

  Vector7d current_velocity = toEigenD<7>(input_parameter.current_velocity);
  auto linear_vel_ref_frame = rot * current_velocity.head<3>();
  auto angular_vel_ref_frame = rot * current_velocity.segment<3>(3);

  Vector7d current_acc = toEigenD<7>(input_parameter.current_acceleration);
  auto linear_acc_ref_frame = rot * current_acc.head<3>();
  auto angular_acc_ref_frame = rot * current_acc.segment<3>(3);

  double elbow_velocity, elbow_acc;
  if (waypoint_has_elbow) {
    elbow_velocity = current_velocity[6];
    elbow_acc = current_acc[6];
  } else {
    elbow_velocity = robot_state.delbow_c;
    elbow_acc = robot_state.ddelbow_c;
  }

  RobotPose zero_pose(Affine::Identity(), current_pose_old_ref_frame.elbow_state());
  Vector7d current_velocity_ref_frame =
      (Vector7d() << linear_vel_ref_frame, angular_vel_ref_frame, elbow_velocity).finished();
  Vector7d current_acc_ref_frame = (Vector7d() << linear_acc_ref_frame, angular_acc_ref_frame, elbow_acc).finished();
  input_parameter.current_position = toStdD<7>(zero_pose.vector_repr());
  input_parameter.current_velocity = toStdD<7>(current_velocity_ref_frame);
  input_parameter.current_acceleration = toStdD<7>(current_acc_ref_frame);

  auto waypoint_pose = new_waypoint.target.pose();

  auto prev_target_robot_pose = target_state_.pose();
  if (!waypoint_pose.elbow_state().has_value()) {
    prev_target_robot_pose = prev_target_robot_pose.withElbowState(ElbowState(robot_state.elbow));
  }

  std::optional<ElbowState> new_elbow;
  if (waypoint_pose.elbow_state().has_value() && new_waypoint.reference_type == ReferenceType::kRelative) {
    if (!prev_target_robot_pose.elbow_state().has_value())
      new_elbow = waypoint_pose.elbow_state();
    else {
      auto new_elbow_pos = waypoint_pose.elbow_state().value().joint_3_pos() +
                           prev_target_robot_pose.elbow_state().value().joint_3_pos();
      new_elbow = ElbowState(new_elbow_pos, waypoint_pose.elbow_state().value().joint_4_flip());
    }
  } else {
    new_elbow = waypoint_pose.elbow_state();
  }
  CartesianState new_target(waypoint_pose.withElbowState(new_elbow), new_waypoint.target.velocity());
  if (new_waypoint.reference_type == ReferenceType::kRelative) {
    new_target = prev_target_robot_pose.end_effector_pose() * new_target;
  }
  auto new_target_transformed = new_target.changeEndEffectorFrame(ee_frame_.inverse());
  auto new_target_ref_frame = ref_frame_.inverse() * new_target_transformed;

  input_parameter.target_position = toStdD<7>(new_target_ref_frame.pose().vector_repr());
  // This is a bit of an oversimplification, as the angular velocities don't
  // work like linear velocities (but we pretend they do here). However, it is
  // probably good enough here.
  input_parameter.target_velocity = toStdD<7>(new_target_ref_frame.velocity().vector_repr());
  input_parameter.enabled = {true, true, true, true, true, true, waypoint_pose.elbow_state().has_value()};

  target_state_ = new_target;
}

std::tuple<Vector7d, Vector7d, Vector7d> CartesianWaypointMotion::getAbsoluteInputLimits() const {
  const auto r = robot();
  return {
      vec_cart_rot_elbow(
          r->translation_velocity_limit.get(), r->rotation_velocity_limit.get(), r->elbow_velocity_limit.get()),
      vec_cart_rot_elbow(
          r->translation_acceleration_limit.get(),
          r->rotation_acceleration_limit.get(),
          r->elbow_acceleration_limit.get()),
      vec_cart_rot_elbow(r->translation_jerk_limit.get(), r->rotation_jerk_limit.get(), r->elbow_jerk_limit.get())};
}

std::tuple<Vector7d, Vector7d, Vector7d> CartesianWaypointMotion::getDesiredState(const RobotState &robot_state) const {
  auto ref_frame_inv = ref_frame_.inverse();
  auto current_pose_ref_frame = ref_frame_inv * RobotPose(robot_state.O_T_EE_d, robot_state.elbow_c);
  auto current_vel_ref_frame = ref_frame_inv * RobotVelocity(robot_state.O_dP_EE_d, robot_state.delbow_c);
  auto current_ee_acc_ref_frame = ref_frame_inv * robot_state.O_ddP_EE_c;
  auto current_elbow_acc = robot_state.ddelbow_c;

  Vector7d current_acc_ref_frame = (Vector7d() << current_ee_acc_ref_frame.vector_repr(), current_elbow_acc).finished();

  return {current_pose_ref_frame.vector_repr(), current_vel_ref_frame.vector_repr(), current_acc_ref_frame};
}

}  // namespace franky
