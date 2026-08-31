#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "franky.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the '_a' literal
using namespace franky;

namespace {

using PyDoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

// A batch of target poses, given either as an (N, 4, 4) array of homogeneous matrices — the
// cheap path, since it costs no per-pose Python objects — or as any sequence of Affine.
std::vector<Affine> toAffines(const py::object &targets) {
  if (py::isinstance<py::array>(targets)) {
    auto array = PyDoubleArray::ensure(targets);
    if (!array || array.ndim() != 3 || array.shape(1) != 4 || array.shape(2) != 4)
      throw std::invalid_argument("o_t_ee array must have shape (N, 4, 4) and a numeric dtype");
    const auto n = static_cast<size_t>(array.shape(0));
    const double *data = array.data();
    std::vector<Affine> result;
    result.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      // The array is C-contiguous, Eigen's default is column-major, hence the RowMajor map.
      result.emplace_back(
          Eigen::Matrix4d{Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(data + 16 * i)});
    }
    return result;
  }
  return targets.cast<std::vector<Affine>>();
}

// Seeds, given as one (7,) configuration shared by every target or as (N, 7).
std::vector<Vector7d> toSeeds(const py::object &q_seed) {
  auto array = PyDoubleArray::ensure(q_seed);
  if (!array || (array.ndim() != 1 && array.ndim() != 2) || array.shape(array.ndim() - 1) != 7)
    throw std::invalid_argument("q_seed must have shape (7,) or (N, 7)");
  const auto n = static_cast<size_t>(array.ndim() == 1 ? 1 : array.shape(0));
  const double *data = array.data();
  std::vector<Vector7d> result;
  result.reserve(n);
  for (size_t i = 0; i < n; ++i) result.emplace_back(Eigen::Map<const Vector7d>(data + 7 * i));
  return result;
}

}  // namespace

void bind_kinematics(py::module &m) {
  py::module kinematics = m.def_submodule("kinematics", "Analytical forward and inverse kinematics.");
  py::enum_<RedundancyParameter>(
      kinematics, "RedundancyParameter", "Free variable used to resolve the Franka arm's 1-DOF redundancy.")
      .value(
          "Swivel",
          RedundancyParameter::kSwivel,
          "Swivel (arm) angle: rotation of the elbow about the shoulder-wrist axis using stereographic SEW. "
          "The intuitive \"elbow\" coordinate; use it when the elbow posture is what you want to specify. Resolved "
          "by a sampled search, so it is far slower than the analytical parameters, holds the "
          "requested angle only approximately, and offers no advantage for path tracking.")
      .value("Q7", RedundancyParameter::kQ7, "Angle of joint 7 (wrist). Fully analytical, exact, and the default.")
      .value(
          "Q6",
          RedundancyParameter::kQ6,
          "Angle of joint 6. Analytical and exact, except in the wrist (type-2) singularity, where the solver "
          "falls back to the Q7 parameterization and cannot hold joint 6 at all; see "
          "IKOptions.redundancy_tolerance.")
      .value(
          "Q5",
          RedundancyParameter::kQ5,
          "Angle of joint 5. Resolved in closed form, so it is analytical and exact like Q6/Q4, with the same "
          "type-2 singularity caveat: there the self-motion does not change joint 5 and the solver falls back "
          "to the Q7 parameterization; see IKOptions.redundancy_tolerance.")
      .value(
          "Q4",
          RedundancyParameter::kQ4,
          "Angle of joint 4 (elbow joint). Analytical and exact, with the same type-2 singularity caveat as Q6.");

  py::class_<IKOptions>(kinematics, "IKOptions", "Options for the analytical inverse kinematics.")
      .def(
          py::init<>([](JointLimits joint_limits,
                        double q1_singularity,
                        double q7_singularity,
                        unsigned int swivel_samples,
                        double pose_tolerance,
                        double redundancy_tolerance) {
            IKOptions options;
            options.joint_limits = std::move(joint_limits);
            options.q1_singularity = q1_singularity;
            options.q7_singularity = q7_singularity;
            options.swivel_samples = swivel_samples;
            options.pose_tolerance = pose_tolerance;
            options.redundancy_tolerance = redundancy_tolerance;
            return options;
          }),
          py::arg_v("joint_limits", kPandaJointLimits, "PANDA_JOINT_LIMITS"),
          "q1_singularity"_a = detail::kPi / 2,
          "q7_singularity"_a = 0.0,
          "swivel_samples"_a = 600,
          "pose_tolerance"_a = 1e-6,
          "redundancy_tolerance"_a = 1e-6)
      .def_readwrite(
          "joint_limits",
          &IKOptions::joint_limits,
          "Joint limits (lower, upper) [rad] used to reject solutions. Defaults to PANDA_JOINT_LIMITS; "
          "set to FR3_JOINT_LIMITS for an FR3.")
      .def_readwrite(
          "q1_singularity",
          &IKOptions::q1_singularity,
          "Fallback value of joint 1 in a shoulder (type-1) singularity [rad].")
      .def_readwrite(
          "q7_singularity",
          &IKOptions::q7_singularity,
          "Fallback value of joint 7 in a wrist (type-2) singularity [rad]. Ignored for Q7.")
      .def_readwrite(
          "swivel_samples",
          &IKOptions::swivel_samples,
          "Number of samples used when resolving the redundancy by swivel angle. Ignored otherwise. Must be in "
          "[MIN_SWIVEL_SAMPLES, MAX_SWIVEL_SAMPLES]; other values raise ValueError. Cost is linear in this value. "
          "A coverage knob rather than an accuracy knob: lowering it mainly costs you targets whose solution the "
          "sweep fails to bracket at all. The typical swivel residual shrinks as samples are added, steeply above "
          "~300 where the solver's local interpolation starts to bite; the worst case does not.")
      .def_readwrite(
          "pose_tolerance",
          &IKOptions::pose_tolerance,
          "Solutions whose forward kinematics misses the target by more than this in position [m] or rotation "
          "angle [rad] are dropped. Guards against the solver's near-singular fallback branches. Costs one fast "
          "forward-kinematics evaluation per candidate; this is a meaningful but minority part of a default solve. "
          "Set to 0 to disable the check.")
      .def_readwrite(
          "redundancy_tolerance",
          &IKOptions::redundancy_tolerance,
          "Solutions whose value for the redundancy parameter differs from the requested one by more than this "
          "[rad, modulo 2*pi] are dropped. The analytical parameters are exact by construction, but in the wrist "
          "(type-2) singularity Q4 and Q6 fall back internally to the Q7 solver, which reaches the pose while "
          "ignoring the requested joint angle entirely -- measured up to 0.70 rad off for Q4 and 0.33 rad for Q6. "
          "pose_tolerance cannot catch that, since the pose itself is correct. Ignored for Swivel, which is "
          "approximate by design. Set to 0 to disable the check.");

  kinematics.attr("PANDA_JOINT_LIMITS") = kPandaJointLimits;
  kinematics.attr("FR3_JOINT_LIMITS") = kFr3JointLimits;
  kinematics.attr("MIN_SWIVEL_SAMPLES") = kMinSwivelSamples;
  kinematics.attr("MAX_SWIVEL_SAMPLES") = kMaxSwivelSamples;

  // These hold the GIL: they are short enough that a release/reacquire pair costs about
  // as much as the call itself. The IK entry points below release conditionally.
  kinematics.def(
      "forward_kinematics",
      &forwardKinematics,
      "q"_a,
      py::arg_v("f_t_ee", Affine::Identity(), "Affine()"),
      "Analytical forward kinematics using the nominal Franka geometry (no robot connection required).\n\n"
      "Args:\n"
      "    q: Joint angles [rad].\n"
      "    f_t_ee: Transform from flange to end-effector frame. Default (identity) returns the flange pose.\n\n"
      "Returns:\n"
      "    The pose of the end-effector frame relative to the base frame.");

  kinematics.def(
      "swivel_angle",
      &swivelAngle,
      "q"_a,
      "The swivel (arm) angle of a configuration [rad] using stereographic SEW. Inverse of resolving the "
      "redundancy by RedundancyParameter.Swivel; use it to hold the current arm posture across an IK call.");

  // A Swivel solve is long enough that holding the GIL across it can push a Python thread
  // running a control or state-polling loop past its deadline; the analytical parameters
  // are short enough that releasing costs more than it saves. py::call_guard would be
  // all-or-nothing, hence the conditional release. Safe here: the solver invokes no
  // Python, and receives its joint limits explicitly through SolverTuning.
  kinematics.def(
      "inverse_kinematics",
      [](const Affine &o_t_ee,
         double redundancy_value,
         RedundancyParameter parameter,
         const Affine &f_t_ee,
         const IKOptions &options) {
        std::array<Vector7d, 8> out_solutions;
        size_t count = 0;
        {
          std::optional<py::gil_scoped_release> release;
          if (parameter == RedundancyParameter::kSwivel) release.emplace();
          count = inverseKinematics(o_t_ee, redundancy_value, parameter, out_solutions, f_t_ee, options);
        }
        py::array_t<double> result({static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(7)});
        if (count > 0) {
          double *dst = result.mutable_data();
          for (size_t i = 0; i < count; ++i) {
            Eigen::Map<Vector7d>(dst + 7 * i) = out_solutions[i];
          }
        }
        return result;
      },
      "o_t_ee"_a,
      "redundancy_value"_a,
      py::arg_v("parameter", RedundancyParameter::kQ7, "RedundancyParameter.Q7"),
      py::arg_v("f_t_ee", Affine::Identity(), "Affine()"),
      "options"_a = IKOptions(),
      "All analytical IK solutions for a target end-effector pose, resolving the redundancy by holding "
      "`parameter` at `redundancy_value`.\n\n"
      "Uses the nominal Franka geometry (no robot connection required). Up to 8 branches; solutions outside "
      "the joint limits, missing the pose, or not holding `redundancy_value` are dropped. The branches are not "
      "guaranteed to be distinct: where two solution families meet, the coincident branch is reported twice.\n\n"
      "Args:\n"
      "    o_t_ee: Target pose of the end-effector frame relative to base.\n"
      "    redundancy_value: Value at which `parameter` is held [rad].\n"
      "    parameter: Which quantity resolves the redundancy.\n"
      "    f_t_ee: Transform from flange to end-effector frame. Default (identity) treats o_t_ee as the flange.\n"
      "    options: Joint limits and singularity/solver options.\n\n"
      "Returns:\n"
      "    An (N, 7) array of valid joint configurations [rad] (where N is 0 to 8).");

  kinematics.def(
      "inverse_kinematics_nearest",
      [](const Affine &o_t_ee,
         const Vector7d &q_seed,
         const Affine &f_t_ee,
         const IKOptions &options,
         std::optional<double>
             redundancy_value,
         RedundancyParameter parameter,
         std::optional<double>
             max_distance) {
        std::optional<py::gil_scoped_release> release;
        if (parameter == RedundancyParameter::kSwivel) release.emplace();
        return inverseKinematicsNearest(o_t_ee, q_seed, f_t_ee, options, redundancy_value, parameter, max_distance);
      },
      "o_t_ee"_a,
      "q_seed"_a,
      py::arg_v("f_t_ee", Affine::Identity(), "Affine()"),
      "options"_a = IKOptions(),
      "redundancy_value"_a = std::nullopt,
      py::arg_v("parameter", RedundancyParameter::kQ7, "RedundancyParameter.Q7"),
      "max_distance"_a = std::nullopt,
      "The single IK solution nearest to `q_seed`, measured as the largest single-joint move.\n\n"
      "If `redundancy_value` is None, the seed's own value for `parameter` is held (e.g. the seed's joint 7), "
      "i.e. the current arm posture is kept.\n\n"
      "Nearest means nearest among the branches that reach the pose within the joint limits, and that set "
      "changes along a Cartesian path, so consecutive calls can switch branches and return a configuration "
      "far from the seed. Pass `max_distance` to get None instead of such a jump.\n\n"
      "Args:\n"
      "    o_t_ee: Target pose of the end-effector frame relative to base.\n"
      "    q_seed: Seed configuration to stay close to [rad].\n"
      "    f_t_ee: Transform from flange to end-effector frame.\n"
      "    options: Joint limits and singularity/solver options.\n"
      "    redundancy_value: Value at which `parameter` is held [rad], or None to hold the seed's value.\n"
      "    parameter: Which quantity resolves the redundancy.\n"
      "    max_distance: Largest acceptable single-joint move from `q_seed` [rad]. If the nearest branch is "
      "farther than this, None is returned instead. None (the default) accepts any distance.\n\n"
      "Returns:\n"
      "    The nearest valid configuration [rad], or None if none exists (or none is within `max_distance`).");

  // Always releases the GIL: one release now covers the whole batch, so unlike the scalar
  // entry points there is no per-solve release to pay for. That is what makes the analytical
  // parameters reachable from more than one core out of Python.
  kinematics.def(
      "inverse_kinematics_nearest_batch",
      [](const py::object &o_t_ee,
         const py::object &q_seed,
         const Affine &f_t_ee,
         const IKOptions &options,
         std::optional<double>
             redundancy_value,
         RedundancyParameter parameter,
         std::optional<double>
             max_distance,
         unsigned int num_threads) {
        std::vector<Affine> targets = toAffines(o_t_ee);
        std::vector<Vector7d> seeds = toSeeds(q_seed);
        const auto n = static_cast<py::ssize_t>(targets.size());

        std::vector<std::optional<Vector7d>> solutions;
        {
          py::gil_scoped_release release;
          solutions = inverseKinematicsNearestBatch(
              targets, seeds, f_t_ee, options, redundancy_value, parameter, max_distance, num_threads);
        }

        py::array_t<double> q({n, static_cast<py::ssize_t>(7)});
        py::array_t<bool> valid(n);
        double *q_data = q.mutable_data();
        bool *valid_data = valid.mutable_data();
        for (py::ssize_t i = 0; i < n; ++i) {
          const std::optional<Vector7d> &solution = solutions[static_cast<size_t>(i)];
          valid_data[i] = solution.has_value();
          // Unsolved targets are filled with NaN, so a caller that ignores the mask gets
          // something that fails loudly rather than a plausible-looking zero configuration.
          Eigen::Map<Vector7d>(q_data + 7 * i) =
              solution.value_or(Vector7d::Constant(std::numeric_limits<double>::quiet_NaN()));
        }
        return py::make_tuple(std::move(q), std::move(valid));
      },
      "o_t_ee"_a,
      "q_seed"_a,
      py::arg_v("f_t_ee", Affine::Identity(), "Affine()"),
      "options"_a = IKOptions(),
      "redundancy_value"_a = std::nullopt,
      py::arg_v("parameter", RedundancyParameter::kQ7, "RedundancyParameter.Q7"),
      "max_distance"_a = std::nullopt,
      "num_threads"_a = 0,
      "`inverse_kinematics_nearest` for many targets at once, solved in parallel.\n\n"
      "Gives the same results as calling `inverse_kinematics_nearest` once per target, in the same order, "
      "but amortizes the per-call overhead and spreads the targets over several threads.\n\n"
      "The targets are solved independently, each from its own seed — never from the previous target's "
      "solution. Seeding each solve with the previous result (the path-tracking idiom) is inherently "
      "sequential and cannot be batched; keep using `inverse_kinematics_nearest` in a loop for that.\n\n"
      "Args:\n"
      "    o_t_ee: Target poses, as an (N, 4, 4) array of homogeneous matrices or a sequence of Affine. The "
      "array form is faster: it costs no per-pose Python objects.\n"
      "    q_seed: Seed configuration [rad], either (7,) to use one seed for every target or (N, 7) for one "
      "seed per target.\n"
      "    f_t_ee: Transform from flange to end-effector frame, shared by all targets.\n"
      "    options: Joint limits and singularity/solver options.\n"
      "    redundancy_value: Value at which `parameter` is held [rad], or None to hold each seed's own value.\n"
      "    parameter: Which quantity resolves the redundancy.\n"
      "    max_distance: Largest acceptable single-joint move from the seed [rad]. Farther branches are "
      "reported as unsolved.\n"
      "    num_threads: Worker threads. 1 forces the single-threaded path; 0 (the default) picks a count from "
      "the machine's core count and the batch size.\n\n"
      "Returns:\n"
      "    A tuple containing a configuration array of shape (N, 7) [rad] and a validity mask of shape (N,). "
      "Rows where valid is False hold NaN and correspond to the None results of "
      "`inverse_kinematics_nearest`.");
}
