#include "franky/kinematics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

#include "franky/util.hpp"
#include "nanogeofik.h"

namespace franky {

// The vendored nanoGeoFIK keeps solver state on the stack. Each call receives
// its joint limits through SolverTuning, so concurrent solves with different
// limits do not need locking.

// Constant rigid transform from the Franka flange to nanoGeoFIK's EndEffector frame
// (the nominal Franka Hand TCP: Rz(-pi/4) then +0.1034 m along z). Computed once
// from nanoGeoFIK itself so we never hard-code the convention here. The static
// initialization is thread-safe (guaranteed by the C++ standard), and franka_fk
// touches no mutable globals, so this is safe to evaluate from any thread.
static const Affine kFlangeToGeofikEE = []() {
  std::array<double, 7> zero{};
  zero.fill(0.0);
  const Eigen::Matrix4d o_t_flange = franka_fk(zero, Frame::Flange);
  const Eigen::Matrix4d o_t_e = franka_fk(zero, Frame::EndEffector);
  return Affine{Eigen::Matrix4d{o_t_flange.inverse() * o_t_e}};
}();

static inline const Affine &flangeToGeofikEE() { return kFlangeToGeofikEE; }

// L-infinity distance between two configurations: the largest single-joint move.
//
// Deliberately not measured modulo 2*pi. No Franka joint has a range wider than 2*pi,
// so a joint angle inside the limits has exactly one valid representative and the
// physical move is the plain difference.
static double distanceLInf(const Vector7d &a, const Vector7d &b) { return (a - b).cwiseAbs().maxCoeff(); }

// A target pose in the form the per-candidate verification wants it: decomposed once per
// solve rather than once per candidate.
//
// Note the .linear() rather than .rotation(), here and everywhere else in this file. Affine
// is Eigen::Affine3d, whose Mode is Affine rather than Isometry, and for that mode
// Transform::rotation() extracts the rotation through a JacobiSVD. Every pose we handle is
// already orthonormal, so the decomposition is a no-op that costs some 25x a plain block
// read -- and the candidate loop ran two of them per branch.
struct TargetPose {
  Eigen::Vector3d position;
  Eigen::Quaterniond rotation;
  double position_tolerance_squared;
  double rotation_tangent_squared;
  bool check_rotation;
};

static TargetPose decompose(const Affine &pose, double tolerance) {
  const bool check_rotation = tolerance < detail::kPi;
  const double tangent = check_rotation ? std::tan(0.5 * tolerance) : 0.0;
  return {
      pose.translation(), Eigen::Quaterniond(pose.linear()), tolerance * tolerance, tangent * tangent, check_rotation};
}

// Whether q reaches the target pose. The solver's near-singular fallback branches can
// return configurations that miss the target by a millimetre or more, so candidates are
// verified before being handed out. The rotation error is taken from the error
// quaternion, which stays accurate near zero (unlike acos of the trace).
static bool reachesPose(const Vector7d &q, const TargetPose &target, const Affine &F_T_EE) {
  const Affine achieved = forwardKinematics(q, F_T_EE);
  if ((achieved.translation() - target.position).squaredNorm() > target.position_tolerance_squared) return false;
  if (!target.check_rotation) return true;
  const Eigen::Quaterniond error = Eigen::Quaterniond(achieved.linear()).conjugate() * target.rotation;
  return error.vec().squaredNorm() <= error.w() * error.w() * target.rotation_tangent_squared;
}

// The value a configuration has for the given redundancy parameter. Used both to read a
// seed's posture and to verify a returned branch.
static double redundancyValue(const Vector7d &q, RedundancyParameter parameter) {
  switch (parameter) {
    case RedundancyParameter::kSwivel:
      return swivelAngle(q);
    case RedundancyParameter::kQ7:
      return q[6];
    case RedundancyParameter::kQ6:
      return q[5];
    case RedundancyParameter::kQ5:
      return q[4];
    case RedundancyParameter::kQ4:
      return q[3];
  }
  return 0.0;
}

// Whether q actually holds the requested redundancy value. The analytical parameters are
// exact by construction, but franka_ik_q4, franka_ik_q5, and franka_ik_q6 fall back internally to
// the q7 solver in the wrist (type-2) singularity, and that fallback reaches the pose while
// ignoring the requested joint angle entirely. reachesPose cannot see it: the pose is
// correct, only the parameter is wrong.
//
// Compared modulo 2*pi. Unlike distanceLInf, which measures a physical move between two
// in-limit configurations, this compares a caller-supplied angle against a solver output,
// and a caller is entitled to name the same wrist angle as q7 or q7 + 2*pi.
//
// Skipped for kSwivel, which is documented as approximate and whose sampled search never
// takes this fallback -- and which would cost a forward kinematics per candidate to check.
static bool holdsRedundancyValue(const Vector7d &q, double requested, RedundancyParameter parameter, double tolerance) {
  if (parameter == RedundancyParameter::kSwivel) return true;
  const double difference = redundancyValue(q, parameter) - requested;
  return std::abs(std::remainder(difference, 2.0 * detail::kPi)) <= tolerance;
}

// clang-format off
const JointLimits kPandaJointLimits{
    (Vector7d() << -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973).finished(),
    (Vector7d() <<  2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973).finished()};

const JointLimits kFr3JointLimits{
    (Vector7d() << -2.7437, -1.7837, -2.9007, -3.0421, -2.8065,  0.5445, -3.0159).finished(),
    (Vector7d() <<  2.7437,  1.7837,  2.9007, -0.1518,  2.8065,  4.5169,  3.0159).finished()};
// clang-format on

Affine forwardKinematics(const Vector7d &q, const Affine &F_T_EE) {
  const Eigen::Matrix4d o_t_flange = franka_fk(toStdD<7>(q), Frame::Flange);
  return Affine{o_t_flange} * F_T_EE;
}

double swivelAngle(const Vector7d &q) { return franka_swivel(toStdD<7>(q)); }

// Reconcile franky's end-effector frame with nanoGeoFIK's fixed EndEffector frame. Right-multiplying a
// target by this yields the pose nanoGeoFIK expects:
//   O_T_geofikE = O_T_EE * (F_T_EE^-1 * F_T_geofikE).
// Depends only on F_T_EE, so a batch sharing one end-effector transform computes it once.
// The Isometry hint spares Eigen a general 3x3 inverse of what is always a rotation.
static Affine eeToGeofik(const Affine &F_T_EE) { return F_T_EE.inverse(Eigen::Isometry) * flangeToGeofikEE(); }

static void validateOptions(RedundancyParameter parameter, const IKOptions &options) {
  if (parameter == RedundancyParameter::kSwivel &&
      (options.swivel_samples < kMinSwivelSamples || options.swivel_samples > kMaxSwivelSamples)) {
    throw std::invalid_argument(
        "IKOptions::swivel_samples must be in [" + std::to_string(kMinSwivelSamples) + ", " +
        std::to_string(kMaxSwivelSamples) + "], got " + std::to_string(options.swivel_samples));
  }
}

static inline SolverTuning makeSolverTuning(
    const IKOptions &options, std::optional<::JointLimits> &custom_limits_storage) {
  SolverTuning tuning;
  tuning.q1_sing = options.q1_singularity;
  tuning.q7_sing = options.q7_singularity;
  tuning.n_points = options.swivel_samples;

  if (options.joint_limits == kPandaJointLimits) {
    tuning.limit_preset = LimitPreset::Panda;
  } else if (options.joint_limits == kFr3JointLimits) {
    tuning.limit_preset = LimitPreset::FR3;
  } else {
    tuning.limit_preset = LimitPreset::Custom;
    custom_limits_storage.emplace(toStdD<7>(options.joint_limits.first), toStdD<7>(options.joint_limits.second));
    tuning.custom_limits = &*custom_limits_storage;
  }
  return tuning;
}

// The body of inverseKinematics, with frame reconciliation and option validation
// already done by the caller, touches no shared mutable state. It is safe to run
// concurrently on distinct targets.
static size_t inverseKinematicsPrepared(
    const Affine &O_T_EE, const Affine &ee_to_geofik, double redundancy_value, RedundancyParameter parameter,
    const Affine &F_T_EE, const IKOptions &options, std::array<Vector7d, 8> &out_solutions) {
  const Affine o_t_geofik_ee = O_T_EE * ee_to_geofik;

  const Eigen::Vector3d position = o_t_geofik_ee.translation();
  const Eigen::Matrix3d rotation = o_t_geofik_ee.linear();
  const std::array<double, 3> r = {position.x(), position.y(), position.z()};
  // nanoGeoFIK expects the rotation matrix in row-first (row-major) order.
  const std::array<double, 9> roe = {
      rotation(0, 0),
      rotation(0, 1),
      rotation(0, 2),
      rotation(1, 0),
      rotation(1, 1),
      rotation(1, 2),
      rotation(2, 0),
      rotation(2, 1),
      rotation(2, 2)};

  std::array<std::array<double, 7>, 8> qsols;

  std::optional<::JointLimits> custom_limits_storage;
  const SolverTuning tuning = makeSolverTuning(options, custom_limits_storage);
  unsigned int candidate_count = 0;
  switch (parameter) {
    case RedundancyParameter::kSwivel:
      candidate_count = franka_ik_swivel(r, roe, redundancy_value, qsols, tuning);
      break;
    case RedundancyParameter::kQ7:
      candidate_count = franka_ik_q7(r, roe, redundancy_value, qsols, tuning);
      break;
    case RedundancyParameter::kQ6:
      candidate_count = franka_ik_q6(r, roe, redundancy_value, qsols, tuning);
      break;
    case RedundancyParameter::kQ5:
      candidate_count = franka_ik_q5(r, roe, redundancy_value, qsols, tuning);
      break;
    case RedundancyParameter::kQ4:
      candidate_count = franka_ik_q4(r, roe, redundancy_value, qsols, tuning);
      break;
  }
  candidate_count = std::min<unsigned int>(candidate_count, qsols.size());

  // Decomposed once, outside the candidate loop: it is the same target for all 8 branches.
  // Do not even construct its quaternion when pose verification is explicitly disabled.
  std::optional<TargetPose> target;
  if (options.pose_tolerance > 0.0) target.emplace(decompose(O_T_EE, options.pose_tolerance));

  size_t valid_count = 0;
  for (unsigned int i = 0; i < candidate_count; ++i) {
    const auto &sol = qsols[i];
    const Vector7d q = toEigenD<7>(sol);
    if (!q.allFinite()) continue;
    if (target.has_value() && !reachesPose(q, *target, F_T_EE)) continue;
    if (options.redundancy_tolerance > 0.0 &&
        !holdsRedundancyValue(q, redundancy_value, parameter, options.redundancy_tolerance))
      continue;
    out_solutions[valid_count++] = q;
  }
  return valid_count;
}

size_t inverseKinematics(
    const Affine &O_T_EE, double redundancy_value, RedundancyParameter parameter,
    std::array<Vector7d, 8> &out_solutions, const Affine &F_T_EE, const IKOptions &options) {
  validateOptions(parameter, options);
  return inverseKinematicsPrepared(
      O_T_EE, eeToGeofik(F_T_EE), redundancy_value, parameter, F_T_EE, options, out_solutions);
}

std::vector<Vector7d> inverseKinematics(
    const Affine &O_T_EE, double redundancy_value, RedundancyParameter parameter, const Affine &F_T_EE,
    const IKOptions &options) {
  std::array<Vector7d, 8> array_sols;
  const size_t count = inverseKinematics(O_T_EE, redundancy_value, parameter, array_sols, F_T_EE, options);
  return std::vector<Vector7d>(array_sols.begin(), array_sols.begin() + count);
}

// The body of inverseKinematicsNearest, sharing inverseKinematicsPrepared's preconditions.
static std::optional<Vector7d> inverseKinematicsNearestPrepared(
    const Affine &O_T_EE, const Affine &ee_to_geofik, const Vector7d &q_seed, const Affine &F_T_EE,
    const IKOptions &options, std::optional<double> redundancy_value, RedundancyParameter parameter,
    std::optional<double> max_distance) {
  const double value = redundancy_value.value_or(redundancyValue(q_seed, parameter));
  const Affine o_t_geofik_ee = O_T_EE * ee_to_geofik;

  const Eigen::Vector3d position = o_t_geofik_ee.translation();
  const Eigen::Matrix3d rotation = o_t_geofik_ee.linear();
  const std::array<double, 3> r = {position.x(), position.y(), position.z()};
  const std::array<double, 9> roe = {
      rotation(0, 0),
      rotation(0, 1),
      rotation(0, 2),
      rotation(1, 0),
      rotation(1, 1),
      rotation(1, 2),
      rotation(2, 0),
      rotation(2, 1),
      rotation(2, 2)};

  std::array<std::array<double, 7>, 8> qsols;

  std::optional<::JointLimits> custom_limits_storage;
  const SolverTuning tuning = makeSolverTuning(options, custom_limits_storage);
  unsigned int candidate_count = 0;
  switch (parameter) {
    case RedundancyParameter::kSwivel:
      candidate_count = franka_ik_swivel(r, roe, value, qsols, tuning);
      break;
    case RedundancyParameter::kQ7:
      candidate_count = franka_ik_q7(r, roe, value, qsols, tuning);
      break;
    case RedundancyParameter::kQ6:
      candidate_count = franka_ik_q6(r, roe, value, qsols, tuning);
      break;
    case RedundancyParameter::kQ5:
      candidate_count = franka_ik_q5(r, roe, value, qsols, tuning);
      break;
    case RedundancyParameter::kQ4:
      candidate_count = franka_ik_q4(r, roe, value, qsols, tuning);
      break;
  }
  candidate_count = std::min<unsigned int>(candidate_count, qsols.size());

  // Rank candidate solutions on the stack without heap allocation
  struct Candidate {
    unsigned int index;
    double distance;
  };
  std::array<Candidate, 8> candidates;
  unsigned int valid_count = 0;

  for (unsigned int i = 0; i < candidate_count; ++i) {
    const auto &sol = qsols[i];
    bool finite = true;
    double d = 0.0;
    for (int j = 0; j < 7; ++j) {
      if (!std::isfinite(sol[j])) {
        finite = false;
        break;
      }
      d = std::max(d, std::abs(sol[j] - q_seed[j]));
    }
    if (finite) {
      if (!max_distance.has_value() || d <= *max_distance) {
        candidates[valid_count++] = {i, d};
      }
    }
  }

  if (valid_count == 0) return std::nullopt;

  // Sort candidate indices by ascending distance to q_seed
  std::sort(candidates.begin(), candidates.begin() + valid_count, [](const Candidate &a, const Candidate &b) {
    return a.distance < b.distance;
  });

  std::optional<TargetPose> target;
  if (options.pose_tolerance > 0.0) target.emplace(decompose(O_T_EE, options.pose_tolerance));

  // Verify only in order of distance. The first candidate passing verification is guaranteed
  // to be the nearest valid solution.
  for (unsigned int i = 0; i < valid_count; ++i) {
    const Vector7d q = toEigenD<7>(qsols[candidates[i].index]);
    if (target.has_value() && !reachesPose(q, *target, F_T_EE)) continue;
    if (options.redundancy_tolerance > 0.0 && !holdsRedundancyValue(q, value, parameter, options.redundancy_tolerance))
      continue;
    return q;
  }

  return std::nullopt;
}

std::optional<Vector7d> inverseKinematicsNearest(
    const Affine &O_T_EE, const Vector7d &q_seed, const Affine &F_T_EE, const IKOptions &options,
    std::optional<double> redundancy_value, RedundancyParameter parameter, std::optional<double> max_distance) {
  validateOptions(parameter, options);
  return inverseKinematicsNearestPrepared(
      O_T_EE, eeToGeofik(F_T_EE), q_seed, F_T_EE, options, redundancy_value, parameter, max_distance);
}

// Worker count for a batch of n targets. An explicit request is honoured (capped at one
// worker per target); the automatic choice additionally refuses to split the batch until
// every worker would get a worthwhile share, below which spawning threads costs more than
// the solves they take over. The share depends on the parameter: a sampled swivel solve is
// some fifty times an analytical one, so it amortizes a thread far sooner.
static unsigned int resolveThreadCount(size_t n, unsigned int requested, RedundancyParameter parameter) {
  if (requested == 1 || n < 2) return 1;
  if (requested > 0) return static_cast<unsigned int>(std::min<size_t>(requested, n));
  const size_t min_per_thread =
      parameter == RedundancyParameter::kSwivel ? kMinSwivelTargetsPerThread : kMinTargetsPerThread;
  // libstdc++ queries the OS on every hardware_concurrency() call. That costs about a
  // microsecond on Linux, enough to matter for the small analytical batches whose
  // thread count is selected here. The process-visible CPU count is effectively stable.
  static const unsigned int available = std::max(1u, std::thread::hardware_concurrency());
  return std::max(1u, static_cast<unsigned int>(std::min<size_t>(n / min_per_thread, available)));
}

std::vector<std::optional<Vector7d>> inverseKinematicsNearestBatch(
    const std::vector<Affine> &targets, const std::vector<Vector7d> &q_seeds, const Affine &F_T_EE,
    const IKOptions &options, std::optional<double> redundancy_value, RedundancyParameter parameter,
    std::optional<double> max_distance, unsigned int num_threads) {
  if (q_seeds.size() != 1 && q_seeds.size() != targets.size()) {
    throw std::invalid_argument(
        "q_seeds must hold either one seed or one per target (" + std::to_string(targets.size()) + "), got " +
        std::to_string(q_seeds.size()));
  }
  validateOptions(parameter, options);

  const size_t n = targets.size();
  std::vector<std::optional<Vector7d>> results(n);
  if (n == 0) return results;

  // Hoisted out of the loop: it is the same for every target, and forcing the lazy static
  // inside flangeToGeofikEE() before any worker starts keeps the workers off its guard.
  const Affine ee_to_geofik = eeToGeofik(F_T_EE);
  const bool one_seed = q_seeds.size() == 1;

  // Distinct indices of a vector of optionals, so the workers never touch the same object.
  auto solve_range = [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      results[i] = inverseKinematicsNearestPrepared(
          targets[i],
          ee_to_geofik,
          one_seed ? q_seeds[0] : q_seeds[i],
          F_T_EE,
          options,
          redundancy_value,
          parameter,
          max_distance);
    }
  };

  const unsigned int threads = resolveThreadCount(n, num_threads, parameter);
  if (threads <= 1) {
    solve_range(0, n);
    return results;
  }

  // The solves themselves do not throw, but their allocations can, and an exception escaping
  // a std::thread would call std::terminate. Each worker keeps its own slot, so recording one
  // needs no lock, and the first is rethrown to the caller after every worker has joined.
  std::vector<std::exception_ptr> errors(threads);
  auto guarded_solve_range = [&](unsigned int slot, size_t begin, size_t end) {
    try {
      solve_range(begin, end);
    } catch (...) {
      errors[slot] = std::current_exception();
    }
  };

  // Static chunking. The per-target cost is near-constant for a given redundancy parameter,
  // so the chunks finish together and work stealing would not earn its complexity.
  std::vector<std::thread> workers;
  workers.reserve(threads - 1);
  const size_t chunk = (n + threads - 1) / threads;
  for (unsigned int t = 1; t < threads; ++t) {
    const size_t begin = std::min(n, t * chunk);
    const size_t end = std::min(n, begin + chunk);
    if (begin < end) workers.emplace_back(guarded_solve_range, t, begin, end);
  }
  guarded_solve_range(0, 0, std::min(n, chunk));
  for (std::thread &worker : workers) worker.join();
  for (const std::exception_ptr &error : errors) {
    if (error) std::rethrow_exception(error);
  }
  return results;
}

}  // namespace franky
