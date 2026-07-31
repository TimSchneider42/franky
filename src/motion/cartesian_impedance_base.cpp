#include "franky/motion/cartesian_impedance_base.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/QR>
#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>
#include <utility>

#include "franky/model.hpp"
#include "franky/motion/torque_control_utils.hpp"

namespace franky {

namespace {

// Rank tolerance for the Jacobian, relative to the largest pivot. Directions below it are dropped
// rather than inverted.
constexpr double kJacobianRankTolerance = 1e-6;

template <typename TaskType>
bool containsTask(const std::vector<NullspaceTask> &tasks) {
  return std::any_of(
      tasks.begin(), tasks.end(), [](const NullspaceTask &task) { return std::holds_alternative<TaskType>(task); });
}

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

Eigen::Matrix<double, 6, 6> computeTaskSpaceInertia(const Jacobian &jacobian, const Eigen::Matrix<double, 7, 7> &mass) {
  const Eigen::Matrix<double, 7, 6> mass_inv_jacobian_transpose = mass.ldlt().solve(jacobian.transpose());
  const Eigen::Matrix<double, 6, 6> task_mass_inv = jacobian * mass_inv_jacobian_transpose;
  // J M^-1 J^T is symmetric positive semidefinite, so its eigenvalues are its singular values and
  // V D^+ V^T comes out symmetric by construction. Eigenvalues also keep their sign, so directions
  // that round negative are dropped
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigensolver(task_mass_inv);
  Eigen::Matrix<double, 6, 6> eigenvalues_inv = Eigen::Matrix<double, 6, 6>::Zero();
  constexpr double tolerance = 1e-6;
  for (int i = 0; i < 6; ++i) {
    if (eigensolver.eigenvalues()[i] > tolerance) eigenvalues_inv(i, i) = 1.0 / eigensolver.eigenvalues()[i];
  }
  return eigensolver.eigenvectors() * eigenvalues_inv * eigensolver.eigenvectors().transpose();
}

Vector7d clampTorque(const Vector7d &tau, std::optional<double> max_torque) {
  if (!max_torque.has_value()) return tau;  // unset ⇒ no clamp
  return tau.cwiseMax(-*max_torque).cwiseMin(*max_torque);
}

// Jacobian quantities shared by the nullspace terms of one control cycle.
struct JacobianNullspaceTerms {
  Eigen::Matrix<double, 7, 6> pinv;

  // Yoshikawa manipulability, the product of the singular values. Zero if rank deficient.
  double manipulability{0.0};
};

JacobianNullspaceTerms computeJacobianNullspaceTerms(const Jacobian &jacobian) {
  Eigen::CompleteOrthogonalDecomposition<Jacobian> cod;
  cod.setThreshold(kJacobianRankTolerance);
  cod.compute(jacobian);

  JacobianNullspaceTerms terms;
  terms.pinv = cod.pseudoInverse();
  // J*P = Q*[T 0]*Z, so det(J J^T) = det(T)^2 and the manipulability is |prod diag(T)|. Forming
  // J J^T would square the condition number and halve the significant digits near a singularity.
  if (cod.rank() == jacobian.rows()) terms.manipulability = std::abs(cod.matrixT().diagonal().prod());
  return terms;
}

// Gradient of the Yoshikawa manipulability, derived entirely from the Jacobian the model already
// supplied. No joint-frame poses and no dJ matrices are required.
//
// For a chain of revolute joints, column k of the geometric Jacobian is v_k = z_k x (p_E - p_k) and
// w_k = z_k. Differentiating with respect to joint i gives
//
//   k <  i:  dv_k/dq_i = z_k x v_i          dw_k/dq_i = 0
//   k >= i:  dv_k/dq_i = z_i x v_k          dw_k/dq_i = z_i x z_k
//
// The k >= i linear term comes out as (z_i x z_k) x r + z_k x (z_i x r), which the Jacobi identity
// collapses to z_i x (z_k x r) = z_i x v_k. So every derivative is one cross product between two
// columns of J, and the joint origins that used to be fetched from the model are exactly the
// information J already carries. This stays valid for arbitrary end-effector and tool offsets
// because it only ever refers to the Jacobian the model returned.
//
// The gradient is w * trace(J^+ dJ_i) = w * sum_k (row k of J^+) . (column k of dJ_i). Splitting
// row k into a_k (linear) and b_k (angular) and applying a . (z x v) = (a x z) . v factors each
// term into a part that does not depend on i:
//
//   k <  i:  a_k . (z_k x v_i)                    = (a_k x z_k) . v_i          =: c_k . v_i
//   k >= i:  a_k . (z_i x v_k) + b_k . (z_i x z_k) = z_i . (v_k x a_k + z_k x b_k) =: z_i . d_k
//
// Neither c_k nor d_k depends on i, so the double loop reduces to a prefix sum over c and a suffix
// sum over d, making the whole gradient linear rather than quadratic in the joint count.
Vector7d manipulabilityGradient(const Jacobian &jacobian, const JacobianNullspaceTerms &terms) {
  const double w = terms.manipulability;
  if (w < 1e-10) return Vector7d::Zero();

  const Eigen::Matrix<double, 7, 6> &J_pinv = terms.pinv;
  const auto v = jacobian.topRows<3>();
  const auto z = jacobian.bottomRows<3>();

  // Gather the pseudoinverse rows into contiguous columns once. Reading them inside the loops
  // instead is a strided access into a column-major matrix and costs more than it saves.
  Eigen::Matrix<double, 3, 7> c;
  Eigen::Matrix<double, 3, 7> d;
  for (int k = 0; k < 7; ++k) {
    const Eigen::Vector3d a = J_pinv.row(k).head<3>().transpose();
    const Eigen::Vector3d b = J_pinv.row(k).tail<3>().transpose();
    c.col(k) = a.cross(z.col(k));
    d.col(k) = v.col(k).cross(a) + z.col(k).cross(b);
  }

  // suffix.col(i) = sum over k >= i of d_k.
  Eigen::Matrix<double, 3, 8> suffix;
  suffix.col(7).setZero();
  for (int i = 6; i >= 0; --i) suffix.col(i) = suffix.col(i + 1) + d.col(i);

  Vector7d gradient;
  Eigen::Vector3d prefix = Eigen::Vector3d::Zero();  // sum over k < i of c_k
  for (int i = 0; i < 7; ++i) {
    gradient[i] = w * (prefix.dot(v.col(i)) + z.col(i).dot(suffix.col(i)));
    prefix += c.col(i);
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
  if ((task.stiffness.array() <= 0.0).all()) return Vector7d::Zero();
  const Vector7d damping = task.damping.value_or(2.0 * task.stiffness.cwiseMax(0.0).cwiseSqrt());
  return clampTorque(
      task.stiffness.cwiseProduct(task.target - robot_state.q) - damping.cwiseProduct(robot_state.dq), task.max_torque);
}

Vector7d computeTaskTorque(
    const ManipulabilityTask &task, const RobotState &robot_state, const Jacobian &jacobian,
    const JacobianNullspaceTerms &terms) {
  if (task.gain == 0.0) return Vector7d::Zero();
  Vector7d tau = task.gain * manipulabilityGradient(jacobian, terms) - task.damping * robot_state.dq;
  return clampTorque(tau, task.max_torque);
}

}  // namespace

CartesianImpedanceBase::CartesianImpedanceBase(
    Affine target, const CartesianImpedanceBase::Params &params, double gains_time_constant)
    : Motion<franka::Torques>(),
      target_(std::move(target)),
      params_(params),
      gains_handle_(CartesianImpedanceGains(params.stiffness, params.damping)),
      nullspace_gains_handle_(nullspaceGainsFromTasks(params.nullspace_tasks)),
      gains_time_constant_(gains_time_constant),
      current_stiffness_(params.stiffness),
      current_nullspace_gains_(nullspaceGainsFromTasks(params.nullspace_tasks)),
      target_gains_(params.stiffness, params.damping),
      target_nullspace_gains_(nullspaceGainsFromTasks(params.nullspace_tasks)),
      has_posture_task_(containsTask<PostureTask>(params.nullspace_tasks)),
      has_manipulability_task_(containsTask<ManipulabilityTask>(params.nullspace_tasks)) {
  if (!std::isfinite(gains_time_constant_) || gains_time_constant_ <= 0.0) {
    throw std::invalid_argument("gains_time_constant must be finite and positive");
  }
  // The real-time path reads target_ with .linear(), so a non-rigid target must be rejected here
  validateIsometry(target_, "target");
  params_.validate();
  critical_damping_ = defaultCartesianImpedanceDamping(current_stiffness_);
  critical_damping_stiffness_ = current_stiffness_;
  current_damping_ = params.damping.value_or(critical_damping_);
}

const Matrix6d &CartesianImpedanceBase::criticalDamping() {
  // Critical damping depends only on the stiffness, and the eigendecomposition is not free on the
  // RT path. Recompute it only while the stiffness is still moving and reuse the cache once settled.
  if (!critical_damping_stiffness_.has_value() ||
      (critical_damping_stiffness_->array() != current_stiffness_.array()).any()) {
    critical_damping_ = defaultCartesianImpedanceDamping(current_stiffness_);
    critical_damping_stiffness_ = current_stiffness_;
  }
  return critical_damping_;
}

franka::Torques CartesianImpedanceBase::computeCommand(
    const RobotState &robot_state, const CartesianReference &reference, double dt) {
  // Interpolate toward the target gains. Gains are published rarely, so re-copy a handle's payload
  // only once the writer has actually pushed one
  if (gains_handle_.hasNewData()) target_gains_ = gains_handle_.getUnsafe();
  const double alpha = 1.0 - std::exp(-dt / gains_time_constant_);
  current_stiffness_ = interpolateGain(current_stiffness_, target_gains_.stiffness, alpha);
  // An unset target means "critically damp the current stiffness"; interpolate toward it like any
  // other gain so unsetting damping is as smooth as setting it. The ternary keeps the
  // eigendecomposition off the explicit-damping path.
  const Matrix6d &target_damping = target_gains_.damping.has_value() ? *target_gains_.damping : criticalDamping();
  current_damping_ += alpha * (target_damping - current_damping_);

  auto &cur = current_nullspace_gains_;
  if (has_posture_task_ || has_manipulability_task_) {
    if (nullspace_gains_handle_.hasNewData()) target_nullspace_gains_ = nullspace_gains_handle_.getUnsafe();
    const NullspaceGains &target_nullspace_gains = target_nullspace_gains_;
    cur.posture_stiffness = interpolateGain(cur.posture_stiffness, target_nullspace_gains.posture_stiffness, alpha);
    const Vector7d posture_critical = 2.0 * cur.posture_stiffness.cwiseMax(0.0).cwiseSqrt();
    cur.posture_damping = interpolateGain(
        cur.posture_damping.value_or(posture_critical),
        target_nullspace_gains.posture_damping.value_or(posture_critical),
        alpha);
    cur.manipulability_gain =
        interpolateGain(cur.manipulability_gain, target_nullspace_gains.manipulability_gain, alpha);
    cur.manipulability_damping =
        interpolateGain(cur.manipulability_damping, target_nullspace_gains.manipulability_damping, alpha);
    // Hard clamp limit (optional). saturateTorqueRate keeps the
    // commanded torque smooth.
    cur.posture_max_torque = target_nullspace_gains.posture_max_torque;
    cur.manipulability_max_torque = target_nullspace_gains.manipulability_max_torque;
  }

  auto model = robot()->model();
  Vector7d coriolis = model->coriolis(robot_state);
  Jacobian jacobian = model->zeroJacobian(franka::Frame::kEndEffector, robot_state);

  Eigen::Quaterniond orientation(robot_state.O_T_EE.linear());

  Eigen::Matrix<double, 6, 1> error;
  error.head(3) << robot_state.O_T_EE.translation() - reference.target.translation();
  error.head(3) = error.head(3).cwiseMax(-params_.translational_error_clip).cwiseMin(params_.translational_error_clip);
  Eigen::Quaterniond quat(reference.target.linear());
  if (quat.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }

  Eigen::Quaterniond error_quaternion(orientation.inverse() * quat);
  const Eigen::AngleAxisd error_angle_axis(error_quaternion);
  const Eigen::Vector3d rotation_vector = error_angle_axis.angle() * error_angle_axis.axis();
  error.tail(3) = -robot_state.O_T_EE.linear() * rotation_vector;
  error.tail(3) = error.tail(3).cwiseMax(-params_.rotational_error_clip).cwiseMin(params_.rotational_error_clip);

  const Vector6d desired_twist =
      reference.target_twist.has_value() ? reference.target_twist->vector_repr() : Vector6d::Zero();
  const Vector6d measured_twist = jacobian * robot_state.dq;

  Vector6d wrench_cartesian = -current_stiffness_ * error - current_damping_ * (measured_twist - desired_twist);
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
  // configured-but-disabled task does not pay for the Jacobian decomposition.
  const bool posture_active = has_posture_task_ && (cur.posture_stiffness.array() > 0.0).any();
  const bool manipulability_active = has_manipulability_task_ && cur.manipulability_gain != 0.0;
  if (posture_active || manipulability_active) {
    const JacobianNullspaceTerms terms = computeJacobianNullspaceTerms(jacobian);
    Vector7d tau_nullspace_unprojected = Vector7d::Zero();
    for (const auto &task : params_.nullspace_tasks) {
      tau_nullspace_unprojected += std::visit(
          [&](const auto &concrete_task) -> Vector7d {
            using Task = std::decay_t<decltype(concrete_task)>;
            const auto effective = applyGains(concrete_task, current_nullspace_gains_);
            if constexpr (std::is_same_v<Task, ManipulabilityTask>) {
              return computeTaskTorque(effective, robot_state, jacobian, terms);
            } else {
              return computeTaskTorque(effective, robot_state);
            }
          },
          task);
    }
    // Orthogonal projector onto the nullspace of J, equal to the textbook I - J^T (J^T)^+ because
    // pinv(J^T) = pinv(J)^T. Applied in factored form so the dense 7x7 projector is never formed.
    tau_nullspace = tau_nullspace_unprojected - terms.pinv * (jacobian * tau_nullspace_unprojected);
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
