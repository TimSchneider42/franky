#include "franky/motion/cartesian_impedance_base.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>
#include <utility>

#include "franky/model.hpp"
#include "franky/motion/torque_control_utils.hpp"

namespace franky {

namespace {

NullspaceGains nullspaceGainsFromTasks(const std::vector<NullspaceTask> &tasks) {
  NullspaceGains gains{};
  for (const auto &task : tasks) {
    std::visit(
        [&](const auto &t) {
          using T = std::decay_t<decltype(t)>;
          if constexpr (std::is_same_v<T, PostureTask>) {
            gains.posture_stiffness = t.stiffness;
            gains.posture_damping = t.damping;
            gains.posture_max_torque = t.max_torque;
          } else {
            gains.manipulability_gain = t.gain;
            gains.manipulability_damping = t.damping;
            gains.manipulability_max_torque = t.max_torque;
          }
        },
        task);
  }
  return gains;
}

double lerp(double current, double target, double alpha) { return current + alpha * (target - current); }

std::optional<double> lerpOptionalDamping(
    std::optional<double> current, std::optional<double> target, double default_current, double alpha) {
  if (!target.has_value()) return std::nullopt;
  return lerp(current.value_or(default_current), *target, alpha);
}
}  // namespace

// Lerp, but snap to target within tolerance so disabling (target 0) reaches the sentinel exactly.
double approach(double current, double target, double alpha) {
  constexpr double kSettleTolerance = 1e-6;
  const double next = current + alpha * (target - current);
  return std::abs(next - target) <= kSettleTolerance ? target : next;
}

Eigen::Matrix<double, 6, 6> computeTaskSpaceInertia(const Jacobian &jacobian, const Eigen::Matrix<double, 7, 7> &mass) {
  const Eigen::Matrix<double, 7, 6> mass_inv_jacobian_transpose = mass.ldlt().solve(jacobian.transpose());
  const Eigen::Matrix<double, 6, 6> task_mass_inv = jacobian * mass_inv_jacobian_transpose;
  Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> svd(task_mass_inv, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix<double, 6, 6> sigma_inv = Eigen::Matrix<double, 6, 6>::Zero();
  constexpr double tolerance = 1e-6;
  for (int i = 0; i < 6; ++i) {
    if (svd.singularValues()[i] > tolerance) sigma_inv(i, i) = 1.0 / svd.singularValues()[i];
  }
  return svd.matrixV() * sigma_inv * svd.matrixU().transpose();
}

Eigen::Matrix<double, 6, 7> pseudoInverse(const Eigen::Matrix<double, 7, 6> &matrix) {
  Eigen::JacobiSVD<Eigen::Matrix<double, 7, 6>> svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const auto &singular_values = svd.singularValues();
  Eigen::Matrix<double, 6, 7> sigma_pinv = Eigen::Matrix<double, 6, 7>::Zero();
  constexpr double tolerance = 1e-6;
  for (int i = 0; i < singular_values.size(); ++i) {
    if (singular_values[i] > tolerance) sigma_pinv(i, i) = 1.0 / singular_values[i];
  }
  return svd.matrixV() * sigma_pinv * svd.matrixU().transpose();
}

Vector7d clampTorque(const Vector7d &tau, std::optional<double> max_torque) {
  if (!max_torque.has_value()) return tau;  // unset ⇒ no clamp
  return tau.cwiseMax(-*max_torque).cwiseMin(*max_torque);
}

double manipulability(const Jacobian &jacobian) {
  const double determinant = (jacobian * jacobian.transpose()).determinant();
  return std::sqrt(std::max(determinant, 0.0));
}

Vector7d manipulabilityGradient(const Model &model, const RobotState &robot_state, const Jacobian &jacobian) {
  const double w = manipulability(jacobian);
  if (w < 1e-10) return Vector7d::Zero();

  Eigen::JacobiSVD<Jacobian> svd(jacobian, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix<double, 7, 6> J_pinv = Eigen::Matrix<double, 7, 6>::Zero();
  constexpr double tolerance = 1e-6;
  for (int i = 0; i < 6; ++i) {
    if (svd.singularValues()[i] > tolerance)
      J_pinv += (1.0 / svd.singularValues()[i]) * svd.matrixV().col(i) * svd.matrixU().col(i).transpose();
  }

  const auto z = jacobian.bottomRows<3>();

  static constexpr std::array<franka::Frame, 7> kJointFrames = {
      franka::Frame::kJoint1,
      franka::Frame::kJoint2,
      franka::Frame::kJoint3,
      franka::Frame::kJoint4,
      franka::Frame::kJoint5,
      franka::Frame::kJoint6,
      franka::Frame::kJoint7};
  Eigen::Matrix<double, 3, 7> p;
  for (int k = 0; k < 7; ++k) p.col(k) = model.pose(kJointFrames[k], robot_state).translation();

  const Eigen::Vector3d pn = robot_state.O_T_EE.translation();

  Vector7d gradient = Vector7d::Zero();
  for (int i = 0; i < 7; ++i) {
    const Eigen::Vector3d zi = z.col(i);
    Eigen::Matrix<double, 6, 7> dJ = Eigen::Matrix<double, 6, 7>::Zero();
    for (int k = 0; k < 7; ++k) {
      const Eigen::Vector3d zk = z.col(k);
      if (k < i) {
        dJ.block<3, 1>(0, k) = zk.cross(zi.cross(pn - p.col(i)));
      } else {
        const Eigen::Vector3d r = pn - p.col(k);
        const Eigen::Vector3d zi_x_zk = zi.cross(zk);
        dJ.block<3, 1>(0, k) = zi_x_zk.cross(r) + zk.cross(zi.cross(r));
        dJ.block<3, 1>(3, k) = zi_x_zk;
      }
    }
    gradient[i] = w * (J_pinv.transpose().array() * dJ.array()).sum();
  }
  return gradient;
}

PostureTask applyGains(PostureTask task, const NullspaceGains &g) {
  task.stiffness = g.posture_stiffness;
  task.damping = g.posture_damping;
  task.max_torque = g.posture_max_torque;
  return task;
}

ManipulabilityTask applyGains(ManipulabilityTask task, const NullspaceGains &g) {
  task.gain = g.manipulability_gain;
  task.damping = g.manipulability_damping;
  task.max_torque = g.manipulability_max_torque;
  return task;
}

Vector7d computeTaskTorque(const PostureTask &task, const RobotState &robot_state) {
  const double stiffness = task.stiffness;
  if (stiffness <= 0.0) return Vector7d::Zero();
  const double damping = task.damping.value_or(2.0 * std::sqrt(stiffness));
  return clampTorque(stiffness * (task.target - robot_state.q) - damping * robot_state.dq, task.max_torque);
}

Vector7d computeTaskTorque(
    const ManipulabilityTask &task, const Model &model, const RobotState &robot_state, const Jacobian &jacobian) {
  if (task.gain == 0.0) return Vector7d::Zero();
  Vector7d tau = task.gain * manipulabilityGradient(model, robot_state, jacobian) - task.damping * robot_state.dq;
  return clampTorque(tau, task.max_torque);
}

}  // namespace

CartesianImpedanceBase::CartesianImpedanceBase(
    Affine target, const CartesianImpedanceBase::Params &params, double gains_time_constant)
    : Motion<franka::Torques>(),
      target_(std::move(target)),
      params_(params),
      gains_handle_(CartesianImpedanceGains(
          params.translational_stiffness, params.rotational_stiffness, params.translational_damping,
          params.rotational_damping)),
      nullspace_gains_handle_(nullspaceGainsFromTasks(params.nullspace_tasks)),
      gains_time_constant_(gains_time_constant),
      current_translational_stiffness_(params.translational_stiffness),
      current_rotational_stiffness_(params.rotational_stiffness),
      current_translational_damping_(params.translational_damping),
      current_rotational_damping_(params.rotational_damping),
      current_nullspace_gains_(nullspaceGainsFromTasks(params.nullspace_tasks)) {
  params_.validate();
  rebuildStiffnessDamping();
}

void CartesianImpedanceBase::rebuildStiffnessDamping() {
  stiffness.setZero();
  stiffness.topLeftCorner(3, 3) << current_translational_stiffness_ * Eigen::MatrixXd::Identity(3, 3);
  stiffness.bottomRightCorner(3, 3) << current_rotational_stiffness_ * Eigen::MatrixXd::Identity(3, 3);
  damping.setZero();
  const double translational_damping =
      current_translational_damping_.value_or(2.0 * std::sqrt(current_translational_stiffness_));
  const double rotational_damping =
      current_rotational_damping_.value_or(2.0 * std::sqrt(current_rotational_stiffness_));
  damping.topLeftCorner(3, 3) << translational_damping * Eigen::MatrixXd::Identity(3, 3);
  damping.bottomRightCorner(3, 3) << rotational_damping * Eigen::MatrixXd::Identity(3, 3);
}

franka::Torques CartesianImpedanceBase::computeCommand(
    const RobotState &robot_state, const CartesianReference &reference, double dt) {
  // Interpolate toward the target gains.
  const auto target_gains = gains_handle_.get();
  const double alpha = 1.0 - std::exp(-dt / gains_time_constant_);
  current_translational_stiffness_ =
      lerp(current_translational_stiffness_, target_gains.translational_stiffness, alpha);
  current_rotational_stiffness_ = lerp(current_rotational_stiffness_, target_gains.rotational_stiffness, alpha);
  current_translational_damping_ = lerpOptionalDamping(
      current_translational_damping_,
      target_gains.translational_damping,
      2.0 * std::sqrt(current_translational_stiffness_),
      alpha);
  current_rotational_damping_ = lerpOptionalDamping(
      current_rotational_damping_,
      target_gains.rotational_damping,
      2.0 * std::sqrt(current_rotational_stiffness_),
      alpha);
  rebuildStiffnessDamping();

  const auto target_nullspace_gains = nullspace_gains_handle_.get();
  auto &cur = current_nullspace_gains_;
  cur.posture_stiffness = approach(cur.posture_stiffness, target_nullspace_gains.posture_stiffness, alpha);
  const double posture_critical = 2.0 * std::sqrt(cur.posture_stiffness);
  cur.posture_damping = approach(
      cur.posture_damping.value_or(posture_critical),
      target_nullspace_gains.posture_damping.value_or(posture_critical),
      alpha);
  cur.manipulability_gain = approach(cur.manipulability_gain, target_nullspace_gains.manipulability_gain, alpha);
  cur.manipulability_damping =
      approach(cur.manipulability_damping, target_nullspace_gains.manipulability_damping, alpha);
  // Hard clamp limit (optional), not a shaped gain: snap it. saturateTorqueRate keeps the
  // commanded torque smooth.
  cur.posture_max_torque = target_nullspace_gains.posture_max_torque;
  cur.manipulability_max_torque = target_nullspace_gains.manipulability_max_torque;

  auto model = robot()->model();
  Vector7d coriolis = model->coriolis(robot_state);
  Jacobian jacobian = model->zeroJacobian(franka::Frame::kEndEffector, robot_state);

  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::Quaterniond orientation(transform.rotation());

  Eigen::Matrix<double, 6, 1> error;
  error.head(3) << robot_state.O_T_EE.translation() - reference.target.translation();
  error.head(3) = error.head(3).cwiseMax(-params_.translational_error_clip).cwiseMin(params_.translational_error_clip);

  Eigen::Quaterniond quat(reference.target.rotation());
  if (quat.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }

  Eigen::Quaterniond error_quaternion(orientation.inverse() * quat);
  error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  error.tail(3) << -transform.linear() * error.tail(3);
  error.tail(3) = error.tail(3).cwiseMax(-params_.rotational_error_clip).cwiseMin(params_.rotational_error_clip);

  const Vector6d desired_twist =
      reference.target_twist.has_value() ? reference.target_twist->vector_repr() : Vector6d::Zero();
  const Vector6d measured_twist = jacobian * robot_state.dq;

  Vector6d wrench_cartesian = -stiffness * error - damping * (measured_twist - desired_twist);
  if (reference.target_acceleration.has_value()) {
    const Eigen::Matrix<double, 7, 7> mass = model->mass(robot_state);
    const Eigen::Matrix<double, 6, 6> lambda = computeTaskSpaceInertia(jacobian, mass);
    wrench_cartesian += lambda * reference.target_acceleration->vector_repr();
  }
  for (int i = 0; i < 6; ++i) {
    if (params_.force_constraints[i].has_value()) wrench_cartesian[i] = *params_.force_constraints[i];
  }

  auto tau_task = jacobian.transpose() * wrench_cartesian;
  Vector7d tau_nullspace = Vector7d::Zero();
  if (!params_.nullspace_tasks.empty()) {
    const auto jacobian_transpose_pinv = pseudoInverse(jacobian.transpose());
    const auto nullspace_projector =
        Eigen::Matrix<double, 7, 7>::Identity() - jacobian.transpose() * jacobian_transpose_pinv;
    Vector7d tau_nullspace_unprojected = Vector7d::Zero();
    for (const auto &task : params_.nullspace_tasks) {
      tau_nullspace_unprojected += std::visit(
          [&](const auto &concrete_task) -> Vector7d {
            using Task = std::decay_t<decltype(concrete_task)>;
            const auto effective = applyGains(concrete_task, current_nullspace_gains_);
            if constexpr (std::is_same_v<Task, ManipulabilityTask>) {
              return computeTaskTorque(effective, *model, robot_state, jacobian);
            } else {
              return computeTaskTorque(effective, robot_state);
            }
          },
          task);
    }
    tau_nullspace = nullspace_projector.transpose() * tau_nullspace_unprojected;
  }

  Vector7d tau_limit = Vector7d::Zero();
  if (params_.safety.lower_joint_limits.has_value() && params_.safety.upper_joint_limits.has_value()) {
    tau_limit = franky::computeJointLimitTorque(
        robot_state.q,
        robot_state.dq,
        *params_.safety.lower_joint_limits,
        *params_.safety.upper_joint_limits,
        params_.safety.joint_limit_activation_distance,
        params_.safety.joint_limit_stiffness,
        params_.safety.joint_limit_damping,
        params_.safety.joint_limit_max_torque);
  }

  Vector7d tau_d = tau_task + tau_nullspace + tau_limit + coriolis;
  tau_d += computeFrictionCompensation(robot_state.dq, params_.friction);
  tau_d = franky::saturateTorqueRate(tau_d, robot_state.tau_J_d, params_.safety.max_delta_tau);

  std::array<double, 7> tau_d_array{};
  Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_d;

  return franka::Torques(tau_d_array);
}

}  // namespace franky
