#ifndef GEOFIK_H
#define GEOFIK_H

#include <Eigen/Dense>
#include <array>
#include <vector>

// NOTE (franky vendoring patch): the upstream header declared `using namespace
// std;` at global scope. That has been removed so the header does not leak the
// std namespace into every translation unit that includes it; the declarations
// below are qualified with std:: instead. The .cpp keeps a file-local `using
// namespace std;` so the implementation is unchanged. See third_party/geofik/README.md.

constexpr double GEOFIK_PI = 3.14159265359;
constexpr double PI = GEOFIK_PI;

// franky vendoring patch: the swivel solvers store their q7 sweep in fixed-size stack
// arrays of this length. Upstream keeps the bound private to the .cpp and does not check
// the caller's n_points against it, so a larger n_points writes past the arrays. Exposing
// it here lets the franky wrapper reject out-of-range values instead. See README.md.
constexpr unsigned int GEOFIK_MAX_N_POINTS = 1000;

// franky vendoring patch: configurable joint limits (see README.md). The upstream
// solver hard-codes the Panda limits as file-scope constants; this setter lets the
// franky wrapper switch them (e.g. to FR3 limits) before solving. q_mid, used by
// check_limits to pick the 2*pi representative of each joint angle, is recomputed
// as the midpoint of [lower, upper]. The limit globals are thread_local (like the
// solver's scratch state), so this sets the limits for the CALLING thread only;
// each thread must configure its own before solving.
void geofik_set_joint_limits(const std::array<double, 7>& lower, const std::array<double, 7>& upper);

/**
 * @brief Computes the joint angles given a Jacobian and the rotation matrix of the ee frame.
 * @param J         transpose of J.
 * @param R         the rotation matrix of ee frame with respect to frame O.
 * @param ee        [optional] name of ee frame ('E', 'F' or '8').
 * @return          joint angles q.
 */
std::array<double, 7> J_to_q(
    const std::array<std::array<double, 6>, 7>& J, const std::array<std::array<double, 3>, 3>& R, const char ee = 'E');

/**
 * @brief Computes the Jacobian given the joint angles.
 * @param q         joint angles, name of ee frame.
 * @param ee        [optional] Name of ee frame ('E', 'F', '8', ...,'1').
 * @return          transpose of J.
 */
std::array<std::array<double, 6>, 7> J_from_q(const std::array<double, 7>& q, const char ee = 'E');

/**
 * @brief Forward kinematics.
 * @param q         joint angles,
 * @param ee        name of ee frame ('E', 'F', '8', ...,'1').
 * @return          transformation matrix of ee frame with respect to frame O.
 */
Eigen::Matrix4d franka_fk(const std::array<double, 7>& q, const char ee = 'E');

/**
 * @brief IK with q7 as free variable.
 * @param r         position of frame E with respect to frame O.
 * @param ROE       rotation matrix of frame E with respect to frame O (row-first format).
 * @param q7        joint angle of joint 7 (radians)
 * @param qsols     array to store 8 solutions
 * @param q1_sing   [optional] emergency value of q1 in case of singularity at shoulder joints (type-1 singularity).
 * @return          number of solutions found.
 */
unsigned int franka_ik_q7(
    const std::array<double, 3>& r, const std::array<double, 9>& ROE, const double q7,
    std::array<std::array<double, 7>, 8>& qsols, const double q1_sing = PI / 2);

/**
 * @brief IK with q4 as free variable.
 * @param r         position of frame E with respect to frame O.
 * @param ROE       rotation matrix of frame E with respect to frame O (row-first format).
 * @param q4        joint angle of joint 4 (radians)
 * @param qsols     array to store 8 solutions
 * @param q1_sing   [optional] emergency value of q1 in case of singularity at shoulder joints (type-1 singularity).
 * @param q7_sing   [optional] emergency value of q7 in case of singularity of S7 intersecting S (type-2 singularity).
 * @return          number of solutions found.
 */
unsigned int franka_ik_q4(
    const std::array<double, 3>& r, const std::array<double, 9>& ROE, const double q4,
    std::array<std::array<double, 7>, 8>& qsols, const double q1_sing = PI / 2, const double q7_sing = 0);

/**
 * @brief IK with q6 as free variable.
 * @param r         position of frame E with respect to frame O.
 * @param ROE       rotation matrix of frame E with respect to frame O (row-first format).
 * @param q6        joint angle of joint 6 (radians)
 * @param qsols     array to store 8 solutions
 * @param q1_sing   [optional] emergency value of q1 in case of singularity at shoulder joints (type-1 singularity).
 * @param q7_sing   [optional] emergency value of q7 in case of singularity of S7 intersecting S (type-2 singularity).
 * @return          number of solutions found.
 */
unsigned int franka_ik_q6(
    const std::array<double, 3>& r, const std::array<double, 9>& ROE, const double q6,
    std::array<std::array<double, 7>, 8>& qsols, const double q1_sing = PI / 2, const double q7_sing = 0);

/**
 * @brief IK with swivel angle as free variable (numerical).
 * @param r         position of frame E with respect to frame O.
 * @param ROE       rotation matrix of frame E with respect to frame O (row-first format).
 * @param theta     swivel angle in radians (see paper for geometric defninition)
 * @param qsols     array to store 8 solutions
 * @param q1_sing   [optional] emergency value of q1 in case of singularity at shoulder joints (type-1 singularity).
 * @param n_points  [optional] number of points to discretise the range of q7.
 * @return          number of solutions found.
 */
unsigned int franka_ik_swivel(
    const std::array<double, 3>& r, const std::array<double, 9>& ROE, const double theta,
    std::array<std::array<double, 7>, 8>& qsols, const double q1_sing = PI / 2, const unsigned int n_points = 500,
    const unsigned int n_fine_search = 3);

/**
 * @brief Calculates the swivel angle given the joint angles q.
 * @param q         joint angles.
 * @return          swivel angle theta (see paper for geometric defninition).
 */
double franka_swivel(const std::array<double, 7>& q);

/**
 * @brief IK to calculate Jacobian and joint angles with q7 as free variable.
 * @param r             position of frame E with respect to frame O.
 * @param ROE           rotation matrix of frame E with respect to frame O (row-first format).
 * @param q7            joint angle of joint 7 (radians).
 * @param Jsols         array to store 8 solutions for the Jacobians.
 * @param qsols         array to store 8 solutions for the joint angles.
 * @param joint_angles  [optional] if false only Jacobians are returned.
 * @param Jacobian_ee   [optional] ee frame of the Jacobian, not the IK ('E', 'F', '8' or '6').
 * @param q1_sing       [optional] emergency value of q1 in case of singularity at shoulder joints (type-1 singularity).
 * @return              number of solutions found.
 */
unsigned int franka_J_ik_q7(
    const std::array<double, 3>& r, const std::array<double, 9>& ROE, const double q7,
    std::array<std::array<std::array<double, 6>, 7>, 8>& Jsols, std::array<std::array<double, 7>, 8>& qsols,
    const bool joint_angles = false, const char Jacobian_ee = 'E', const double q1_sing = PI / 2);

/**
 * @brief IK to calculate Jacobian and joint angles with q4 as free variable.
 * @param r             position of frame E with respect to frame O.
 * @param ROE           rotation matrix of frame E with respect to frame O (row-first format).
 * @param q4            joint angle of joint 4 (radians).
 * @param Jsols         array to store 8 solutions for the Jacobians.
 * @param qsols         array to store 8 solutions for the joint angles.
 * @param joint_angles  [optional] if false only Jacobians are returned.
 * @param Jacobian_ee   [optional] ee frame of the Jacobian, not the IK ('E', 'F', '8' or '6').
 * @param q1_sing       [optional] emergency value of q1 in case of singularity at shoulder joints (type-1 singularity).
 * @param q7_sing       [optional] emergency value of q7 in case of singularity of S7 intersecting S (type-2
 * singularity).
 * @return              number of solutions found.
 */
unsigned int franka_J_ik_q4(
    const std::array<double, 3>& r, const std::array<double, 9>& ROE, const double q4,
    std::array<std::array<std::array<double, 6>, 7>, 8>& Jsols, std::array<std::array<double, 7>, 8>& qsols,
    const bool joint_angles = false, const char Jacobian_ee = 'E', const double q1_sing = PI / 2,
    const double q7_sing = 0);

/**
 * @brief IK to calculate Jacobian and joint angles with q6 as free variable.
 * @param r             position of frame E with respect to frame O.
 * @param ROE           rotation matrix of frame E with respect to frame O (row-first format).
 * @param q6            joint angle of joint 6 (radians).
 * @param Jsols         array to store 8 solutions for the Jacobians.
 * @param qsols         array to store 8 solutions for the joint angles.
 * @param joint_angles  [optional] if false only Jacobians are returned.
 * @param Jacobian_ee   [optional] ee frame of the Jacobian, not the IK ('E', 'F', '8' or '6').
 * @param q1_sing       [optional] emergency value of q1 in case of singularity at shoulder joints (type-1 singularity).
 * @param q7_sing       [optional] emergency value of q7 in case of singularity of S7 intersecting S (type-2
 * singularity).
 * @return              number of solutions found.
 */
unsigned int franka_J_ik_q6(
    const std::array<double, 3>& r, const std::array<double, 9>& ROE, const double q6,
    std::array<std::array<std::array<double, 6>, 7>, 8>& Jsols, std::array<std::array<double, 7>, 8>& qsols,
    const bool joint_angles = false, const char Jacobian_ee = 'E', const double q1_sing = PI / 2,
    const double q7_sing = 0);

/**
 * @brief IK to calculate Jacobian and joint angles with swivel angle as free variable (numerical).
 * @param r             position of frame E with respect to frame O.
 * @param ROE           rotation matrix of frame E with respect to frame O (row-first format).
 * @param theta         swivel angle in radians (see paper for geometric defninition).
 * @param Jsols         array to store 8 solutions for the Jacobians.
 * @param qsols         array to store 8 solutions for the joint angles.
 * @param joint_angles  [optional] if false only Jacobians are returned.
 * @param Jacobian_ee   [optional] ee frame of the Jacobian, not the IK ('E', 'F', '8' or '6').
 * @param q1_sing       [optional] emergency value of q1 in case of singularity at shoulder joints (type-1 singularity).
 * @param n_points      [optional] number of points to discretise the range of q7.
 * @return              number of solutions found.
 */
unsigned int franka_J_ik_swivel(
    const std::array<double, 3>& r, const std::array<double, 9>& ROE, const double theta,
    std::array<std::array<std::array<double, 6>, 7>, 8>& Jsols, std::array<std::array<double, 7>, 8>& qsols,
    const bool joint_angles = false, const char Jacobian_ee = 'E', const double q1_sing = PI / 2,
    const unsigned int n_points = 600, const unsigned int n_fine_search = 3);

#endif
