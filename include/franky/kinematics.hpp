#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "franky/types.hpp"

namespace franky {

namespace detail {
inline constexpr double kPi = 3.14159265358979323846;
}  // namespace detail

/**
 * @brief The free variable used to resolve the Franka arm's 1-DOF redundancy.
 *
 * The Franka arm is kinematically redundant: for a given end-effector pose there
 * is a one-parameter family of joint configurations. These options select which
 * quantity is held fixed to pick a configuration out of that family.
 */
enum class RedundancyParameter {
  /**
   * The swivel (arm) angle: the rotation of the elbow about the axis connecting
   * the shoulder and the wrist, parameterized using stereographic SEW (Elias &
   * Wen 2024). The intuitive "elbow" coordinate; use it when the elbow posture is
   * what you want to specify. Resolved by a sampled search rather than in closed
   * form, so it is far slower than the parameters below and holds the requested
   * angle only approximately (see IKOptions::swivel_samples). It offers no
   * advantage for following a Cartesian path.
   */
  kSwivel,
  /** The angle of joint 7 (the wrist). Fully analytical, exact, and the default. */
  kQ7,
  /**
   * The angle of joint 6. Analytical and exact, except in the wrist (type-2)
   * singularity, where the solver falls back to the joint-7 parameterization and
   * cannot hold joint 6 at all; see IKOptions::redundancy_tolerance.
   */
  kQ6,
  /**
   * The angle of joint 5. Resolved in closed form (a quartic in the wrist-circle
   * angle), so it is exact and fully analytical like kQ6/kQ4, with the same
   * type-2 singularity caveat: when axis 7 passes through the shoulder the
   * self-motion does not change joint 5, and the solver hands the problem to
   * the joint-7 parameterization; see IKOptions::redundancy_tolerance.
   */
  kQ5,
  /**
   * The angle of joint 4 (the elbow joint). Analytical and exact, with the same
   * type-2 singularity caveat as kQ6.
   */
  kQ4
};

/**
 * @brief Joint position limits as a {lower, upper} pair [rad].
 */
using JointLimits = std::pair<Vector7d, Vector7d>;

/**
 * @brief Joint position limits of the Franka Emika Panda (and FER) [rad].
 */
extern const JointLimits kPandaJointLimits;

/**
 * @brief Joint position limits of the Franka Research 3 (FR3) [rad].
 */
extern const JointLimits kFr3JointLimits;

/**
 * @brief Largest usable value of IKOptions::swivel_samples.
 *
 * The underlying solver sweeps joint 7 into fixed-size buffers of this length.
 * inverseKinematics throws std::invalid_argument for larger values.
 */
inline constexpr unsigned int kMaxSwivelSamples = 1000;

/**
 * @brief Smallest usable value of IKOptions::swivel_samples.
 */
inline constexpr unsigned int kMinSwivelSamples = 2;

/**
 * @brief Options for the analytical inverse kinematics.
 */
struct IKOptions {
  /**
   * @brief Joint limits used to reject solutions. Solutions with any joint
   * outside these limits are dropped, and each joint angle is wrapped to the
   * 2*pi representative closest to the center of its range. Defaults to the
   * Panda limits; set to kFr3JointLimits for an FR3.
   */
  JointLimits joint_limits = kPandaJointLimits;

  /**
   * @brief Fallback value of joint 1 used when the arm is in a shoulder
   * (type-1) singularity, where joint 1 is otherwise undetermined [rad].
   */
  double q1_singularity = detail::kPi / 2;

  /**
   * @brief Fallback value of joint 7 used in a wrist (type-2) singularity
   * [rad]. Ignored when the redundancy parameter is kQ7.
   */
  double q7_singularity = 0.0;

  /**
   * @brief Number of samples used to discretize the search when resolving the
   * redundancy by swivel angle. Ignored for the analytical parameters
   * (kQ7 / kQ6 / kQ5 / kQ4).
   *
   * Must lie in [kMinSwivelSamples, kMaxSwivelSamples]; other values throw
   * std::invalid_argument. Cost is linear in this value.
   *
   * Treat it as a coverage knob rather than an accuracy knob: lowering it mainly
   * costs you targets whose solution the sweep fails to bracket at all. The typical
   * residual in the held swivel angle does shrink as samples are added -- steeply
   * above ~300 samples, where the solver's local interpolation starts to bite -- but
   * the worst case does not, since it is set by the targets the sweep never brackets.
   */
  unsigned int swivel_samples = 600;

  /**
   * @brief Tolerance for verifying that a solution actually reaches the target
   * pose: candidates whose forward kinematics deviates by more than this in
   * position [m] or in rotation angle [rad] are dropped.
   *
   * Near singularities the solver's fallback branches can return configurations
   * that miss the target by a millimetre or more, so every candidate is checked
   * before it is returned. Set to 0 (or a negative value) to disable it and return
   * every branch the solver produced.
   *
   * The check costs one fast forward-kinematics evaluation per candidate. It is a
   * meaningful part of an analytical solve, but substantially cheaper than the
   * solve itself. Disable it only if you are verifying the result some other way.
   */
  double pose_tolerance = 1e-6;

  /**
   * @brief Tolerance for verifying that a solution actually holds the requested
   * redundancy value [rad]: candidates whose value for the redundancy parameter
   * differs from the requested one by more than this are dropped. Compared modulo
   * 2*pi, so asking for a 2*pi-equivalent angle is not penalised.
   *
   * The analytical parameters are exact by construction, but in the wrist (type-2)
   * singularity -- the target frame's z-axis pointing near the shoulder -- kQ4, kQ5, and
   * kQ6 fall back internally to the kQ7 solver, which reaches the pose while ignoring
   * the requested joint angle entirely (measured up to 0.70 rad off for kQ4 and
   * 0.33 rad for kQ6). IKOptions::pose_tolerance cannot catch that, because the pose
   * is correct; only the redundancy parameter is wrong.
   *
   * The default has wide margin on both sides: over 400k random reachable targets the
   * genuine residual stayed below 3e-13 rad, while every singular fallback exceeded
   * 1e-3 rad. Ignored for RedundancyParameter::kSwivel, which is documented as
   * approximate and is resolved by a search that never takes this fallback. Set to 0
   * (or a negative value) to disable the check.
   */
  double redundancy_tolerance = 1e-6;
};

/**
 * @brief Analytical forward kinematics using the Franka arm's nominal geometry.
 *
 * Unlike Model::pose, this does not require a franka::Model (and hence no robot
 * connection), which makes it usable offline. It uses the nominal Franka DH
 * parameters, so it does not account for per-robot calibration.
 *
 * @param q      Joint angles [rad].
 * @param F_T_EE Transformation from the flange to the end-effector frame. The
 *               default (identity) returns the flange pose.
 * @return The pose of the end-effector frame relative to the base frame.
 */
[[nodiscard]] Affine forwardKinematics(const Vector7d &q, const Affine &F_T_EE = Affine::Identity());

/**
 * @brief The swivel (arm) angle of a joint configuration [rad] using stereographic SEW.
 *
 * This is the inverse of resolving the redundancy by RedundancyParameter::kSwivel:
 * it maps a configuration to the swivel value that reproduces it. Useful for
 * holding the current arm posture across an IK call.
 *
 * @param q Joint angles [rad].
 * @return The swivel angle [rad] in [-pi, pi].
 */
[[nodiscard]] double swivelAngle(const Vector7d &q);

/**
 * @brief Computes all analytical IK solutions for a target end-effector pose.
 *
 * The redundancy is resolved by holding @p parameter fixed at @p redundancy_value.
 * The Franka arm admits up to 8 solution branches; solutions violating the joint
 * limits in @p options, failing to reproduce @p O_T_EE within
 * IKOptions::pose_tolerance, or failing to hold @p redundancy_value within
 * IKOptions::redundancy_tolerance, are dropped, so the returned vector may contain
 * anywhere from 0 to 8 configurations. The branches are not guaranteed to be
 * distinct: where two solution families meet -- near full arm extension, or where the
 * two wrist roots coincide -- the solver reports the coincident branch twice.
 *
 * With RedundancyParameter::kSwivel the redundancy is resolved by a sampled search, so
 * the swivel angle of the returned configurations only approximates @p redundancy_value,
 * and the solve is far slower. The returned configurations reach @p O_T_EE either way.
 *
 * This uses the nominal Franka geometry (no per-robot calibration) and does not
 * require a robot connection.
 *
 * @param O_T_EE           Target pose of the end-effector frame relative to base.
 * @param redundancy_value Value at which @p parameter is held [rad].
 * @param parameter        Which quantity resolves the redundancy.
 * @param F_T_EE           Transformation from flange to end-effector frame. The
 *                         default (identity) treats @p O_T_EE as the flange pose.
 * @param options          Joint limits and singularity/solver options.
 * @return The valid joint configurations [rad].
 * @throw std::invalid_argument if IKOptions::swivel_samples is out of range and
 *        @p parameter is RedundancyParameter::kSwivel.
 */
[[nodiscard]] std::vector<Vector7d> inverseKinematics(
    const Affine &O_T_EE, double redundancy_value, RedundancyParameter parameter = RedundancyParameter::kQ7,
    const Affine &F_T_EE = Affine::Identity(), const IKOptions &options = {});

/**
 * @brief Zero-allocation overload of inverseKinematics for real-time control loops.
 *
 * Writes valid joint configurations into @p out_solutions and returns the number of solutions found.
 * Guaranteed not to perform heap allocations.
 *
 * @param O_T_EE           Target pose of the end-effector frame relative to base.
 * @param redundancy_value Value at which @p parameter is held [rad].
 * @param parameter        Which quantity resolves the redundancy.
 * @param out_solutions    Fixed-size array (length 8) to receive valid joint configurations.
 * @param F_T_EE           Transformation from flange to end-effector frame.
 * @param options          Joint limits and singularity/solver options.
 * @return Number of valid joint configurations written into @p out_solutions (0 to 8).
 */
[[nodiscard]] size_t inverseKinematics(
    const Affine &O_T_EE, double redundancy_value, RedundancyParameter parameter,
    std::array<Vector7d, 8> &out_solutions, const Affine &F_T_EE = Affine::Identity(), const IKOptions &options = {});

/**
 * @brief Computes the single IK solution nearest to a seed configuration.
 *
 * Enumerates the branches (as inverseKinematics) and returns the one closest to
 * @p q_seed under an L-infinity (largest single-joint) distance. This is the
 * trajectory-tracking / "reach this pose from where I am" idiom.
 *
 * "Nearest" is only nearest among the branches that reach the pose within the
 * configured limits, and that set changes along a Cartesian path: consecutive calls
 * can switch branches and hand back a configuration far from the seed. No choice of
 * @p parameter avoids this. Pass @p max_distance to turn such a jump into an explicit
 * std::nullopt instead of a command the robot will try to execute.
 *
 * @param O_T_EE           Target pose of the end-effector frame relative to base.
 * @param q_seed           Seed configuration to stay close to [rad].
 * @param F_T_EE           Transformation from flange to end-effector frame.
 * @param options          Joint limits and singularity/solver options.
 * @param redundancy_value Value at which @p parameter is held [rad]. If not set,
 *                         the seed's own value for @p parameter is used (e.g. the
 *                         seed's joint 7), i.e. the current arm posture is held.
 * @param parameter        Which quantity resolves the redundancy.
 * @param max_distance     If set, the largest single-joint move from @p q_seed that
 *                         is acceptable [rad]. When the nearest branch is farther
 *                         than this, std::nullopt is returned rather than that
 *                         branch. Unset (the default) accepts any distance.
 * @return The nearest valid configuration, or std::nullopt if none exists (or none
 *         is within @p max_distance).
 */
[[nodiscard]] std::optional<Vector7d> inverseKinematicsNearest(
    const Affine &O_T_EE, const Vector7d &q_seed, const Affine &F_T_EE = Affine::Identity(),
    const IKOptions &options = {}, std::optional<double> redundancy_value = std::nullopt,
    RedundancyParameter parameter = RedundancyParameter::kQ7, std::optional<double> max_distance = std::nullopt);

/**
 * @brief Smallest number of targets given to a worker thread when
 * inverseKinematicsNearestBatch picks the thread count itself and the redundancy is
 * resolved analytically (kQ7 / kQ6 / kQ5 / kQ4).
 *
 * A batch is split only while every worker still gets at least this many targets.
 * Measured crossovers vary from 8 to 24 targets with the solver and CPU topology; this
 * conservative shared threshold first splits at 32 targets.
 */
inline constexpr size_t kMinTargetsPerThread = 16;

/**
 * @brief The kMinTargetsPerThread equivalent for RedundancyParameter::kSwivel.
 *
 * A sampled swivel solve costs some fifty times an analytical one, which dwarfs the cost
 * of spawning a thread; splitting already pays with two targets.
 */
inline constexpr size_t kMinSwivelTargetsPerThread = 1;

/**
 * @brief Solves inverseKinematicsNearest for many targets at once.
 *
 * Equivalent to calling inverseKinematicsNearest once per entry of @p targets, but it
 * hoists the per-call setup out of the loop and may spread the targets over several
 * threads. The results are identical to the per-target calls, in the same order.
 *
 * The targets are solved independently: a target's seed is @p q_seeds[i], never the
 * solution of the previous target. Seeding each solve with the previous result — the
 * path-tracking idiom — is inherently sequential and cannot be batched; keep calling
 * inverseKinematicsNearest in a loop for that.
 *
 * @param targets          Target poses of the end-effector frame relative to base.
 * @param q_seeds          Seed configurations [rad]. Either one seed, applied to every
 *                         target, or exactly one per target.
 * @param F_T_EE           Transformation from flange to end-effector frame, shared by
 *                         all targets.
 * @param options          Joint limits and singularity/solver options.
 * @param redundancy_value Value at which @p parameter is held [rad]. If not set, each
 *                         solve holds its own seed's value, as in inverseKinematicsNearest.
 * @param parameter        Which quantity resolves the redundancy.
 * @param max_distance     If set, the largest single-joint move from the seed that is
 *                         acceptable [rad]; farther branches yield std::nullopt.
 * @param num_threads      Worker threads to use. 1 forces the single-threaded path. 0
 *                         (the default) picks a count from the hardware concurrency and
 *                         the batch size, never giving a worker fewer than
 *                         kMinTargetsPerThread targets (kMinSwivelTargetsPerThread for
 *                         RedundancyParameter::kSwivel).
 * @return One result per target, std::nullopt where inverseKinematicsNearest would
 *         return std::nullopt.
 * @throw std::invalid_argument if @p q_seeds is neither of size 1 nor of size
 *        targets.size(), or if IKOptions::swivel_samples is out of range and @p parameter
 *        is RedundancyParameter::kSwivel.
 */
[[nodiscard]] std::vector<std::optional<Vector7d>> inverseKinematicsNearestBatch(
    const std::vector<Affine> &targets, const std::vector<Vector7d> &q_seeds, const Affine &F_T_EE = Affine::Identity(),
    const IKOptions &options = {}, std::optional<double> redundancy_value = std::nullopt,
    RedundancyParameter parameter = RedundancyParameter::kQ7, std::optional<double> max_distance = std::nullopt,
    unsigned int num_threads = 0);

}  // namespace franky
