#include "franky/motion/cartesian_waypoint_motion.hpp"

#include <ruckig/ruckig.hpp>
#include <utility>

#include "franky/cartesian_state.hpp"
#include "franky/robot.hpp"
#include "franky/util.hpp"

namespace franky {

CartesianWaypointMotion::CartesianWaypointMotion(const std::vector<Waypoint<CartesianState>> &waypoints)
    : CartesianWaypointMotion(waypoints, Params()) {}

CartesianWaypointMotion::CartesianWaypointMotion(const std::vector<Waypoint<CartesianState>> &waypoints, Params params)
    : params_(std::move(params)),
      WaypointMotion<franka::CartesianPose, CartesianState>(waypoints, params) {}

void CartesianWaypointMotion::initWaypointMotion(
    const franka::RobotState &robot_state,
    const std::optional<franka::CartesianPose> &previous_command,
    ruckig::InputParameter<7> &input_parameter) {
  RobotPose robot_pose(previous_command.value_or(franka::CartesianPose{robot_state.O_T_EE_c, robot_state.elbow_c}));
  ref_frame_ = Affine::Identity();

  auto initial_velocity = RobotVelocity(franka::CartesianVelocities{robot_state.O_dP_EE_c, robot_state.delbow_c});

  auto initial_acceleration = Vector6d::Map(robot_state.O_ddP_EE_c.data());
  Vector7d initial_acceleration_with_elbow = (Vector7d() << initial_acceleration, robot_state.ddelbow_c[0]).finished();

  target_state_ = robot_pose;

  input_parameter.current_position = toStd<7>(robot_pose.vector_repr());
  input_parameter.current_velocity = toStd<7>(initial_velocity.vector_repr());
  input_parameter.current_acceleration = toStd<7>(initial_acceleration_with_elbow);
}

franka::CartesianPose CartesianWaypointMotion::getControlSignal(
    const ruckig::InputParameter<7> &input_parameter) const {
  auto has_elbow = input_parameter.enabled[6];
  return (ref_frame_ * RobotPose(toEigen<7>(input_parameter.current_position), !has_elbow)).as_franka_pose();
}

void CartesianWaypointMotion::setNewWaypoint(
    const franka::RobotState &robot_state,
    const std::optional<franka::CartesianPose> &previous_command,
    const Waypoint<CartesianState> &new_waypoint,
    ruckig::InputParameter<7> &input_parameter) {
  auto waypoint_has_elbow = input_parameter.enabled[6];

  // We first convert the current state into the frame of the current pose
  RobotPose current_pose_old_ref_frame = RobotPose(toEigen<7>(input_parameter.current_position), !waypoint_has_elbow);
  Affine new_ref_to_old_ref = current_pose_old_ref_frame.end_effector_pose();
  ref_frame_ = ref_frame_ * new_ref_to_old_ref;
  auto rot = new_ref_to_old_ref.inverse().rotation();

  Vector7d current_velocity = toEigen<7>(input_parameter.current_velocity);
  auto linear_vel_ref_frame = rot * current_velocity.head<3>();
  auto angular_vel_ref_frame = rot * current_velocity.segment<3>(3);

  Vector7d current_acc = toEigen<7>(input_parameter.current_acceleration);
  auto linear_acc_ref_frame = rot * current_acc.head<3>();
  auto angular_acc_ref_frame = rot * current_acc.segment<3>(3);

  double elbow_velocity, elbow_acc;
  if (waypoint_has_elbow) {
    elbow_velocity = current_velocity[6];
    elbow_acc = current_acc[6];
  } else {
    elbow_velocity = robot_state.delbow_c[6];
    elbow_acc = robot_state.ddelbow_c[6];
  }

  RobotPose zero_pose(Affine::Identity(), current_pose_old_ref_frame.elbow_position());
  Vector7d current_velocity_ref_frame =
      (Vector7d() << linear_vel_ref_frame, angular_vel_ref_frame, elbow_velocity).finished();
  Vector7d current_acc_ref_frame =
      (Vector7d() << linear_acc_ref_frame, angular_acc_ref_frame, elbow_acc).finished();
  input_parameter.current_position = toStd<7>(zero_pose.vector_repr());
  input_parameter.current_velocity = toStd<7>(current_velocity_ref_frame);
  input_parameter.current_acceleration = toStd<7>(current_acc_ref_frame);

  auto waypoint_pose = new_waypoint.target.pose();

  auto prev_target_robot_pose = target_state_.pose();
  if (!waypoint_pose.elbow_position().has_value()) {
    prev_target_robot_pose = prev_target_robot_pose.withElbowPosition(robot_state.elbow[0]);
  }

  std::optional<double> new_elbow;
  if (waypoint_pose.elbow_position().has_value() && new_waypoint.reference_type == ReferenceType::Relative) {
    if (!prev_target_robot_pose.elbow_position().has_value())
      new_elbow = waypoint_pose.elbow_position();
    else {
      new_elbow = waypoint_pose.elbow_position().value() + prev_target_robot_pose.elbow_position().value();
    }
  } else {
    new_elbow = waypoint_pose.elbow_position();
  }
  CartesianState new_target(waypoint_pose.withElbowPosition(new_elbow), new_waypoint.target.velocity());
  if (new_waypoint.reference_type == ReferenceType::Relative) {
    new_target = prev_target_robot_pose.end_effector_pose() * new_target;
  }
  auto new_target_transformed = new_target.changeEndEffectorFrame(params_.frame.inverse());
  auto new_target_ref_frame = ref_frame_.inverse() * new_target_transformed;

  input_parameter.target_position = toStd<7>(new_target_ref_frame.pose().vector_repr());
  // This is a bit of an oversimplification, as the angular velocities don't work like linear velocities (but we pretend
  // they do here). However, it is probably good enough here.
  input_parameter.target_velocity = toStd<7>(new_target_ref_frame.velocity().vector_repr());
  input_parameter.enabled = {true, true, true, true, true, true, waypoint_pose.elbow_position().has_value()};

  target_state_ = new_target;
}

std::tuple<Vector7d, Vector7d, Vector7d>
CartesianWaypointMotion::getAbsoluteInputLimits() const {
  constexpr double translation_factor{0.4};
  constexpr double derivative_factor{0.4};

  Vector7d max_vel = vec_cart_rot_elbow(
      translation_factor * Robot::max_translation_velocity,
      Robot::max_rotation_velocity,
      Robot::max_elbow_velocity);
  Vector7d max_acc = derivative_factor * vec_cart_rot_elbow(
      translation_factor * Robot::max_translation_acceleration,
      Robot::max_rotation_acceleration,
      Robot::max_elbow_acceleration);
  Vector7d max_jerk = std::pow(derivative_factor, 2) * vec_cart_rot_elbow(
      translation_factor * Robot::max_translation_jerk,
      Robot::max_rotation_jerk,
      Robot::max_elbow_jerk);
  return {max_vel, max_acc, max_jerk};
}

}  // namespace franky
