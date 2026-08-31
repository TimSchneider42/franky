// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Pablo Lopez-Custodio, 2026 Nick Walker

#pragma once

#include <Eigen/Dense>
#include <array>
#include <vector>

constexpr double PI = 3.14159265358979323846;

enum class Frame : uint8_t {
  EndEffector = 0,  // Hand TCP / panda_hand_tcp (formerly '''E''')
  Flange,  // Link 8 / Flange / panda_link8 (formerly '''F''' / '''8''')
  Link7,   // Link 7 / panda_link7
  Link6,   // Link 6 / panda_link6
  Link5,   // Link 5 / panda_link5
  Link4,   // Link 4 / panda_link4
  Link3,   // Link 3 / panda_link3
  Link2,   // Link 2 / panda_link2
  Link1,   // Link 1 / panda_link1

  // Aliases
  EE = EndEffector,
  TCP = EndEffector,
};

/**
 * @brief What a franka_J_ik_* solver writes to its outputs.
 *
 * Required, never defaulted: it decides whether `qsols` is meaningful on
 * return. Under Output::JacobianOnly the joint angles are left as NaN, which
 * is a silent trap if it was not what you meant.
 */
enum class Output : uint8_t {
  JacobianOnly = 0,       // Fill Jsols; leave qsols filled with NaN.
  JointsAndJacobian = 1,  // Fill both.
};

/**
 * @brief Robot model / limit preset for compile-time limit optimization.
 */
enum class LimitPreset : uint8_t {
  Panda = 0,  // Franka Emika Panda / FER limits (compile-time optimized)
  FR3,        // Franka Research 3 limits (compile-time optimized)
  Custom,     // User-supplied custom limits from SolverTuning::custom_limits
  None,       // Skip joint limit checks and clipping entirely
};

/**
 * @brief Joint position limits for all 7 Franka joints [rad].
 */
struct JointLimits {
  std::array<double, 7> lower{};
  std::array<double, 7> upper{};
  std::array<double, 7> middle{};

  constexpr JointLimits() = default;
  constexpr JointLimits(const std::array<double, 7>& l,
                        const std::array<double, 7>& u)
      : lower(l), upper(u), middle{0.5 * (l[0] + u[0]), 0.5 * (l[1] + u[1]),
                                   0.5 * (l[2] + u[2]), 0.5 * (l[3] + u[3]),
                                   0.5 * (l[4] + u[4]), 0.5 * (l[5] + u[5]),
                                   0.5 * (l[6] + u[6])} {}
  constexpr JointLimits(const std::array<double, 7>& l,
                        const std::array<double, 7>& u,
                        const std::array<double, 7>& m)
      : lower(l), upper(u), middle(m) {}
};

constexpr JointLimits kPandaJointLimits = {
    {-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973},
    {2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973},
    {0.0, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0}};

constexpr JointLimits kFr3JointLimits = {
    {-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159},
    {2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159},
    {0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307, 0.0}};

/**
 * @brief Rarely-touched solver parameters, shared by every entry point so the
 * signatures stay uniform. Each solver uses the subset that applies to it:
 *
 *   q1_sing        every solver
 *   q7_sing        q4, q5, q6 (q7 and swivel parameterise q7 directly)
 *   n_points,
 *   n_fine_search  swivel only
 *   limit_preset,
 *   custom_limits,
 *   check_joint_limits  every solver
 *
 * The defaults are the values the solvers used before this struct existed,
 * with one deliberate change: franka_J_ik_swivel() previously defaulted
 * n_points to 600 while franka_ik_swivel() used 500, so the same query got a
 * different sweep resolution depending on whether you asked for Jacobians.
 * Both now use 500.
 */
struct SolverTuning {
  double q1_sing = PI / 2;  // fallback q1 at a shoulder (type-1) singularity
  double q7_sing = 0.0;     // fallback q7 at a type-2 singularity
  unsigned int n_points = 500;     // swivel coarse sweep points
  unsigned int n_fine_search = 3;  // swivel refinement steps

  LimitPreset limit_preset = LimitPreset::Panda;
  const JointLimits* custom_limits = nullptr;
  bool check_joint_limits = true;
};

// ============================================================================
// Joint-only Inverse Kinematics Solvers
// ============================================================================

/**
 * @brief IK with q4 as free variable.
 * @param r         position of frame E with respect to frame O.
 * @param ROE       rotation matrix of frame E with respect to frame O
 * (row-first format).
 * @param q4        joint angle of joint 4 (radians).
 * @param qsols     std::array to store 8 solutions.
 * @param tuning    [optional] singularity fallbacks and sweep resolution.
 * @return          number of solutions found.
 */
unsigned int franka_ik_q4(const std::array<double, 3>& r,
                          const std::array<double, 9>& ROE, const double q4,
                          std::array<std::array<double, 7>, 8>& qsols,
                          const SolverTuning& tuning = {});

/**
 * @brief IK with q5 as free variable.
 * @param r         position of frame E with respect to frame O.
 * @param ROE       rotation matrix of frame E with respect to frame O
 * (row-first format).
 * @param q5        joint angle of joint 5 (radians).
 * @param qsols     std::array to store 8 solutions.
 * @param tuning    [optional] singularity fallbacks and sweep resolution.
 * @return          number of solutions found.
 */
unsigned int franka_ik_q5(const std::array<double, 3>& r,
                          const std::array<double, 9>& ROE, const double q5,
                          std::array<std::array<double, 7>, 8>& qsols,
                          const SolverTuning& tuning = {});

/**
 * @brief IK with q6 as free variable.
 * @param r         position of frame E with respect to frame O.
 * @param ROE       rotation matrix of frame E with respect to frame O
 * (row-first format).
 * @param q6        joint angle of joint 6 (radians).
 * @param qsols     std::array to store 8 solutions.
 * @param tuning    [optional] singularity fallbacks and sweep resolution.
 * @return          number of solutions found.
 */
unsigned int franka_ik_q6(const std::array<double, 3>& r,
                          const std::array<double, 9>& ROE, const double q6,
                          std::array<std::array<double, 7>, 8>& qsols,
                          const SolverTuning& tuning = {});

/**
 * @brief IK with q7 as free variable.
 * @param r         position of frame E with respect to frame O.
 * @param ROE       rotation matrix of frame E with respect to frame O
 * (row-first format).
 * @param q7        joint angle of joint 7 (radians).
 * @param qsols     std::array to store 8 solutions.
 * @param tuning    [optional] singularity fallbacks and sweep resolution.
 * @return          number of solutions found.
 */
unsigned int franka_ik_q7(const std::array<double, 3>& r,
                          const std::array<double, 9>& ROE, const double q7,
                          std::array<std::array<double, 7>, 8>& qsols,
                          const SolverTuning& tuning = {});

/**
 * @brief IK with swivel angle as free variable using stereographic SEW.
 * @param r         position of frame E with respect to frame O.
 * @param ROE       rotation matrix of frame E with respect to frame O
 * (row-first format).
 * @param theta     swivel angle in radians (stereographic SEW
 * parameterization).
 * @param qsols     std::array to store 8 solutions.
 * @param tuning    [optional] singularity fallbacks and sweep resolution.
 * @return          number of solutions found.
 */
unsigned int franka_ik_swivel(const std::array<double, 3>& r,
                              const std::array<double, 9>& ROE,
                              const double theta,
                              std::array<std::array<double, 7>, 8>& qsols,
                              const SolverTuning& tuning = {});

// ============================================================================
// Joint + Jacobian Inverse Kinematics Solvers
//
//   output       required; Output::JointsAndJacobian to fill qsols as well as
//                Jsols, Output::JacobianOnly to skip the joint-angle recovery.
//                Never defaulted -- under JacobianOnly qsols comes back NaN.
//   Jacobian_ee  frame the Jacobian is expressed in.
// ============================================================================

/**
 * @brief IK with q4 as free variable, computing Jacobians and, under
 * Output::JointsAndJacobian, joint angles.
 */
unsigned int franka_J_ik_q4(
    const std::array<double, 3>& r, const std::array<double, 9>& ROE,
    const double q4, std::array<std::array<std::array<double, 6>, 7>, 8>& Jsols,
    std::array<std::array<double, 7>, 8>& qsols, const Output output,
    const Frame Jacobian_ee = Frame::EndEffector,
    const SolverTuning& tuning = {});

/**
 * @brief IK with q5 as free variable, computing Jacobians and, under
 * Output::JointsAndJacobian, joint angles.
 */
unsigned int franka_J_ik_q5(
    const std::array<double, 3>& r, const std::array<double, 9>& ROE,
    const double q5, std::array<std::array<std::array<double, 6>, 7>, 8>& Jsols,
    std::array<std::array<double, 7>, 8>& qsols, const Output output,
    const Frame Jacobian_ee = Frame::EndEffector,
    const SolverTuning& tuning = {});

/**
 * @brief IK with q6 as free variable, computing Jacobians and, under
 * Output::JointsAndJacobian, joint angles.
 */
unsigned int franka_J_ik_q6(
    const std::array<double, 3>& r, const std::array<double, 9>& ROE,
    const double q6, std::array<std::array<std::array<double, 6>, 7>, 8>& Jsols,
    std::array<std::array<double, 7>, 8>& qsols, const Output output,
    const Frame Jacobian_ee = Frame::EndEffector,
    const SolverTuning& tuning = {});

/**
 * @brief IK with q7 as free variable, computing Jacobians and, under
 * Output::JointsAndJacobian, joint angles.
 */
unsigned int franka_J_ik_q7(
    const std::array<double, 3>& r, const std::array<double, 9>& ROE,
    const double q7, std::array<std::array<std::array<double, 6>, 7>, 8>& Jsols,
    std::array<std::array<double, 7>, 8>& qsols, const Output output,
    const Frame Jacobian_ee = Frame::EndEffector,
    const SolverTuning& tuning = {});

/**
 * @brief IK with swivel angle as free variable, computing Jacobians and, under
 * Output::JointsAndJacobian, joint angles.
 */
unsigned int franka_J_ik_swivel(
    const std::array<double, 3>& r, const std::array<double, 9>& ROE,
    const double theta,
    std::array<std::array<std::array<double, 6>, 7>, 8>& Jsols,
    std::array<std::array<double, 7>, 8>& qsols, const Output output,
    const Frame Jacobian_ee = Frame::EndEffector,
    const SolverTuning& tuning = {});

// ============================================================================
// Kinematics and Jacobian Utilities
// ============================================================================

/**
 * @brief Forward kinematics.
 * @param q         joint angles.
 * @param ee        end-effector frame (defaults to Frame::EndEffector).
 * @return          transformation matrix of ee frame with respect to frame O.
 */
Eigen::Matrix4d franka_fk(const std::array<double, 7>& q,
                          const Frame ee = Frame::EndEffector);

/**
 * @brief Computes the Jacobian given the joint angles.
 * @param q         joint angles.
 * @param ee        [optional] end-effector frame (defaults to
 * Frame::EndEffector).
 * @return          transpose of J.
 */
std::array<std::array<double, 6>, 7> J_from_q(
    const std::array<double, 7>& q, const Frame ee = Frame::EndEffector);

/**
 * @brief Computes the joint angles given a Jacobian and the rotation matrix of
 * the ee frame.
 * @param J         transpose of J.
 * @param R         the rotation matrix of ee frame with respect to frame O.
 * @param ee        [optional] end-effector frame (Frame::EndEffector or
 * Frame::Flange).
 * @return          joint angles q.
 */
std::array<double, 7> J_to_q(const std::array<std::array<double, 6>, 7>& J,
                             const std::array<std::array<double, 3>, 3>& R,
                             const Frame ee = Frame::EndEffector);

/**
 * @brief Calculates the swivel angle given the joint angles q using
 * stereographic SEW.
 * @param q         joint angles.
 * @return          swivel angle theta in radians ([-pi, pi]).
 */
double franka_swivel(const std::array<double, 7>& q);
