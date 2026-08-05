/**
 * @file    geofik.cpp
 * @brief   functions for the IK of the Franka arm.
 *
 * @details notation:
 *          - `r_PQ_O`: position vector of point P with respect to point Q measured in frame O.
 *          - `ROF`: Rotation matrix representing the orientation of a frame F with respect to frame O.
 *
 * @author  Pablo Lopez-Custodio
 * @date    2025-02-08
 * @version 1.0
 */

#include "geofik.h"

#include <cmath>
#include <iostream>

// franky vendoring patch: the upstream header carried a global `using namespace
// std;`. It was removed from the header (to avoid leaking std into every
// includer) and moved here, so the implementation below compiles unchanged.
using namespace std;

// franky vendoring patch: upstream writes solver diagnostics to std::cout. franky is a
// robot control library, so writing to stdout from a solver is not acceptable. The messages therefore go to a null
// sink, or to std::cerr when GEOFIK_VERBOSE is defined. See README.md.
#ifdef GEOFIK_VERBOSE
static std::ostream& geofik_log() { return std::cerr; }
#else
static std::ostream& geofik_log() {
  static struct NullBuffer : std::streambuf {
    int overflow(int c) override { return c; }
  } buffer;
  static std::ostream stream(&buffer);
  return stream;
}
#endif

constexpr double d1 = 0.333;
constexpr double d3 = 0.316;
constexpr double a4 = 0.0825;
constexpr double a5 = 0.0825;
constexpr double d5 = 0.384;
constexpr double a7 = 0.088;
// dE =  0.107 + 0.1034
constexpr double dE = 0.2104;
// b1 = sqrt(d3*d3 + a4*a4)
constexpr double b1 = 0.3265918706887849;
// b2 = sqrt(d5*d5 + a5*a5)
constexpr double b2 = 0.39276233271534583;
// beta1 = arctan(a4/d3)
constexpr double beta1 = 0.25537561488738186;
// beta2 = arctan(a5/d5)
constexpr double beta2 = 0.21162680876562978;

// Tolerance for entering singularity mode.
//
// franky vendoring patch: upstream compared *squared* quantities -- squared cross-product
// norms at the type-2 singularity gates, squared deviations from +-1 at the acos/asin domain
// clamps -- against a bare SING_TOL in some places and against SING_TOL*SING_TOL in others,
// so nominally identical tests differed by a factor of 316 in the underlying quantity. All
// squared comparisons now use SING_TOL*SING_TOL; bare comparisons are against plain
// (unsquared) norms, where SING_TOL is the intended tolerance. See README.md.
constexpr double SING_TOL = 1e-5;

// The shoulder fallback divides by the horizontal projection of the third joint
// axis. Double precision remains well-conditioned far closer to zero than the
// geometric assembly tests above; using their 1e-5 tolerance here needlessly
// replaced valid near-singular solutions with an arbitrary q1 value.
constexpr double SHOULDER_SING_TOL = 1e-8;

// The closed-form q6 construction contains 1/sin(q6), so use its parallel
// fallback only in the narrow region where that construction is genuinely
// ill-conditioned. Reusing SING_TOL here created an avoidable no-solution band
// for requested q6 values between roughly 1e-6 and 1e-5 rad.
constexpr double WRIST_PARALLEL_TOL = 1e-6;

// Inverse-trig domain clamps are for floating-point overshoot only. A looser
// geometric tolerance admits roots that do not assemble into the requested arm.
constexpr double TRIG_DOMAIN_TOL = 1e-8;

// A reconstructed joint angle can land a few ulps outside an inclusive hardware
// limit even when the source configuration lies exactly on that limit. Keep the
// tolerance far below any mechanically meaningful angle and clamp only that
// round-off-sized overshoot back to the boundary.
constexpr double JOINT_LIMIT_TOL = 1e-10;

// error threshold for swivel angle solver
// # define ERR_THRESH 0.01 // this slightly smaller than 1deg
// franky vendoring patch: upstream squared the swivel error and used 0.002 here
// ("slightly smaller than (3deg)*(3deg)"). The error is no longer squared (see
// theta_err_from_q7), so this is the same acceptance band expressed as an angle:
// sqrt(0.002) = 0.0447 rad, slightly under 3 deg. See README.md.
constexpr double ERR_THRESH = 0.0447;  // slightly smaller than 3deg
// max number of points in discretisation for swivel angle solver
const unsigned int MAX_N_POINTS = GEOFIK_MAX_N_POINTS;

// franky vendoring patch: these were `const` in upstream and hard-coded to the
// Panda joint limits. They are now mutable and settable via geofik_set_joint_limits
// so the franky wrapper can switch to FR3 limits. Defaults remain the Panda values,
// so behavior is identical to upstream unless the setter is called. q_mid is the
// per-joint center used by check_limits to pick the 2*pi representative of an angle.
//
// They are `thread_local` (see the scratch globals below) so each thread carries
// its own limit configuration; a thread must call geofik_set_joint_limits itself to
// change its limits. This lets concurrent threads solve safely without locking.
thread_local array<double, 7> q_low = {-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973};
// franky vendoring patch: q_up[1] read 1.762 upstream, a typo for the Panda's 1.7628. It
// never bit franky (the wrapper calls geofik_set_joint_limits before every solve), but it
// is wrong for anyone reading these defaults. See README.md.
thread_local array<double, 7> q_up = {2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973};
thread_local array<double, 7> q_mid = {0.0, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0};

void geofik_set_joint_limits(const array<double, 7>& lower, const array<double, 7>& upper) {
  for (int i = 0; i < 7; i++) {
    q_low[i] = lower[i];
    q_up[i] = upper[i];
    q_mid[i] = 0.5 * (lower[i] + upper[i]);
  }
}

// Jacobian at home configuration (orientation only)
const Eigen::Matrix<double, 3, 7> J0_S(
    {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0}, {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0}});

// franky vendoring patch: helper functions R_axis_angle and J_dir are pure functions
// returning stack-allocated Eigen matrices rather than writing to file-scope global/TLS
// temporaries. This makes all solver functions threadsafe and reentrant without TLS overhead.

Eigen::Matrix3d R_axis_angle(const Eigen::Vector3d& s, double theta) {
  double x = s[0];
  double y = s[1];
  double z = s[2];
  double ct = cos(theta);
  double st = sin(theta);
  double one_minus_ct = 1 - ct;
  Eigen::Matrix3d R;
  R << ct + x * x * one_minus_ct, x * y * one_minus_ct - z * st, x * z * one_minus_ct + y * st,
      y * x * one_minus_ct + z * st, ct + y * y * one_minus_ct, y * z * one_minus_ct - x * st,
      z * x * one_minus_ct - y * st, z * y * one_minus_ct + x * st, ct + z * z * one_minus_ct;
  return R;
}

Eigen::Matrix3d R_axis_angle(const array<double, 3>& s, double theta) {
  double x = s[0];
  double y = s[1];
  double z = s[2];
  double ct = cos(theta);
  double st = sin(theta);
  double one_minus_ct = 1 - ct;
  Eigen::Matrix3d R;
  R << ct + x * x * one_minus_ct, x * y * one_minus_ct - z * st, x * z * one_minus_ct + y * st,
      y * x * one_minus_ct + z * st, ct + y * y * one_minus_ct, y * z * one_minus_ct - x * st,
      z * x * one_minus_ct - y * st, z * y * one_minus_ct + x * st, ct + z * z * one_minus_ct;
  return R;
}

array<double, 3> Cross(const Eigen::Vector3d& u, const Eigen::Vector3d& v) {
  return array<double, 3>{u[1] * v[2] - v[1] * u[2], v[0] * u[2] - u[0] * v[2], u[0] * v[1] - v[0] * u[1]};
}

array<double, 3> Cross(const array<double, 3>& u, const array<double, 3>& v) {
  return array<double, 3>{u[1] * v[2] - v[1] * u[2], v[0] * u[2] - u[0] * v[2], u[0] * v[1] - v[0] * u[1]};
}

array<double, 3> Cross(const array<double, 3>& u, const Eigen::Vector3d& v) {
  return array<double, 3>{u[1] * v[2] - v[1] * u[2], v[0] * u[2] - u[0] * v[2], u[0] * v[1] - v[0] * u[1]};
}

array<double, 3> Cross(const Eigen::Vector3d& u, const array<double, 3>& v) {
  return array<double, 3>{u[1] * v[2] - v[1] * u[2], v[0] * u[2] - u[0] * v[2], u[0] * v[1] - v[0] * u[1]};
}

void Cross_(const array<double, 3>& u, const Eigen::Vector3d& v, array<double, 3>& w) {
  w[0] = u[1] * v[2] - v[1] * u[2];
  w[1] = v[0] * u[2] - u[0] * v[2];
  w[2] = u[0] * v[1] - v[0] * u[1];
}

void Cross_(const array<double, 3>& u, const array<double, 3>& v, array<double, 3>& w) {
  w[0] = u[1] * v[2] - v[1] * u[2];
  w[1] = v[0] * u[2] - u[0] * v[2];
  w[2] = u[0] * v[1] - v[0] * u[1];
}

double Dot(const Eigen::Vector3d& u, const Eigen::Vector3d& v) { return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]; }

double Dot(const array<double, 3>& u, const array<double, 3>& v) { return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]; }

double Dot(const array<double, 3>& u, const Eigen::Vector3d& v) { return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]; }

double Dot(const Eigen::Vector3d& u, const array<double, 3>& v) { return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]; }

double Norm(const Eigen::Vector3d& u) { return sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]); }

double Norm(const array<double, 3>& u) { return sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]); }

Eigen::Matrix<double, 3, 7> J_dir(
    const array<double, 3>& s2, const array<double, 3>& s3, const array<double, 3>& s4, const array<double, 3>& s5,
    const array<double, 3>& s6, const array<double, 3>& s7) {
  Eigen::Matrix<double, 3, 7> J;
  J << 0, s2[0], s3[0], s4[0], s5[0], s6[0], s7[0], 0, s2[1], s3[1], s4[1], s5[1], s6[1], s7[1], 1, s2[2], s3[2],
      s4[2], s5[2], s6[2], s7[2];
  return J;
}

void save_J_sol(
    const array<double, 3>& s2, const array<double, 3>& s3, const array<double, 3>& s4, const array<double, 3>& s5,
    const array<double, 3>& s6, const array<double, 3>& s7, const array<double, 3>& r4, const array<double, 3>& r5,
    const array<double, 3>& r_EO_O, array<array<array<double, 6>, 7>, 8>& Jsols, const int index,
    const char Jacobian_ee) {
  // saves the two Jacobian solutions for the given joint axes at Jsols[2*index] and Jsols[2*index+1].
  // Jacobian_ee is the frame of the Jacobian end-effector ('6', '8', 'F' or 'E')
  // r4 = r_4S_O
  // r5 = r_5S_O
  array<double, 3> r_1ee_O, r_4ee_O, r_5ee_O;
  if (Jacobian_ee == '6') {
    // r_P6_O = r_PS_O + r_S6_O
    //          r_PS_O - r_6S_O remember r_6S_O = r_5S_O
    r_1ee_O = {-r5[0], -r5[1], -r5[2]};
    r_4ee_O = {r4[0] - r5[0], r4[1] - r5[1], r4[2] - r5[2]};
    r_5ee_O = {0, 0, 0};
  } else if (Jacobian_ee == '8' || Jacobian_ee == 'F') {
    // r_PF_O = r_PE_O + r_EF_O
    //        = r_PS_O - rEO_O + rSO_O + r_EF_O
    //        = r_PS_O - rEO_O + (0,0,d1) + 0.1034*s7_O
    r_1ee_O = {-r_EO_O[0] + 0.1034 * s7[0], -r_EO_O[1] + 0.1034 * s7[1], d1 - r_EO_O[2] + 0.1034 * s7[2]};
    r_4ee_O = {
        r4[0] - r_EO_O[0] + 0.1034 * s7[0],
        r4[1] - r_EO_O[1] + 0.1034 * s7[1],
        d1 + r4[2] - r_EO_O[2] + 0.1034 * s7[2]};
    r_5ee_O = {
        r5[0] - r_EO_O[0] + 0.1034 * s7[0],
        r5[1] - r_EO_O[1] + 0.1034 * s7[1],
        d1 + r5[2] - r_EO_O[2] + 0.1034 * s7[2]};
  } else {
    // r_PE_O = r_PS_O + r_SE_O
    //        = r_PS_O - r_ES_O
    //        = r_PS_O - (r_EO_O + r_OS_O) = r_PS_O - r_EO_O + r_SO_O
    r_1ee_O = {-r_EO_O[0], -r_EO_O[1], d1 - r_EO_O[2]};
    r_4ee_O = {r4[0] - r_EO_O[0], r4[1] - r_EO_O[1], d1 + r4[2] - r_EO_O[2]};
    r_5ee_O = {r5[0] - r_EO_O[0], r5[1] - r_EO_O[1], d1 + r5[2] - r_EO_O[2]};
  }

  array<double, 3> m;
  Jsols[2 * index][0] = {0, 0, 1, r_1ee_O[1], -r_1ee_O[0], 0};  // r_1ee_O x (0,0,1) = (r_1ee_O[1], -r_1ee_O[0], 0)
  Jsols[2 * index + 1][0] = {0, 0, 1, r_1ee_O[1], -r_1ee_O[0], 0};
  Cross_(r_1ee_O, s2, m);  // r_2ee_O = r_1ee_O
  Jsols[2 * index][1] = {s2[0], s2[1], s2[2], m[0], m[1], m[2]};
  Jsols[2 * index + 1][1] = {-s2[0], -s2[1], -s2[2], -m[0], -m[1], -m[2]};  // second solution of spherical shoulder
  Cross_(r_1ee_O, s3, m);                                                   //  r3_ee = r1_ee
  Jsols[2 * index][2] = {s3[0], s3[1], s3[2], m[0], m[1], m[2]};
  Jsols[2 * index + 1][2] = {s3[0], s3[1], s3[2], m[0], m[1], m[2]};
  Cross_(r_4ee_O, s4, m);
  Jsols[2 * index][3] = {s4[0], s4[1], s4[2], m[0], m[1], m[2]};
  Jsols[2 * index + 1][3] = {s4[0], s4[1], s4[2], m[0], m[1], m[2]};
  Cross_(r_5ee_O, s5, m);
  Jsols[2 * index][4] = {s5[0], s5[1], s5[2], m[0], m[1], m[2]};
  Jsols[2 * index + 1][4] = {s5[0], s5[1], s5[2], m[0], m[1], m[2]};
  Cross_(r_5ee_O, s6, m);  // r6 = r5
  Jsols[2 * index][5] = {s6[0], s6[1], s6[2], m[0], m[1], m[2]};
  Jsols[2 * index + 1][5] = {s6[0], s6[1], s6[2], m[0], m[1], m[2]};
  if (Jacobian_ee == '6') {
    Jsols[2 * index][6] = {0, 0, 0, 0, 0, 0};
    Jsols[2 * index + 1][6] = {0, 0, 0, 0, 0, 0};
  } else {
    Jsols[2 * index][6] = {s7[0], s7[1], s7[2], 0, 0, 0};
    Jsols[2 * index + 1][6] = {s7[0], s7[1], s7[2], 0, 0, 0};
  }
}

double signed_angle(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, const Eigen::Vector3d& s) {
  return atan2(Dot(Cross(v1, v2), s), Dot(v1, v2));
}

double signed_angle(const array<double, 3>& v1, const array<double, 3>& v2, const array<double, 3>& s) {
  // return atan2(s[2]*(v1[0]*v2[1] - v1[1]*v2[0]) - s[1]*(v1[0]*v2[2] - v1[2]*v2[0]) + s[0]*(v1[1]*v2[2] -
  // v1[2]*v2[1]), v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]);
  return atan2(Dot(Cross(v1, v2), s), Dot(v1, v2));
}

// Both inputs must be wrapped to [-pi, pi], so their difference needs at most
// one turn adjustment. This replaces the much more expensive acos(cos(a - b))
// identity while retaining the sign needed by the swivel solver's secant step.
static double wrapped_angle_difference(const double a, const double b) {
  double difference = a - b;
  if (difference > PI)
    difference -= 2 * PI;
  else if (difference < -PI)
    difference += 2 * PI;
  return difference;
}

void check_limits(array<double, 7>& q, int n) {
  for (int i = 0; i < n; i++) {
    q[i] = q_mid[i] + atan2(sin(q[i] - q_mid[i]), cos(q[i] - q_mid[i]));
    if (q[i] < q_low[i])
      q[i] = q[i] >= q_low[i] - JOINT_LIMIT_TOL ? q_low[i] : NAN;
    else if (q[i] > q_up[i])
      q[i] = q[i] <= q_up[i] + JOINT_LIMIT_TOL ? q_up[i] : NAN;
  }
}

// whole rotational part of J
array<double, 6> q_from_J(const Eigen::Matrix<double, 3, 7>& J) {
  const int dof = 7;
  array<double, dof - 1> q;
  Eigen::Matrix<double, 3, 7> J_old = J0_S;  // [3,dof]
  for (int i = 0; i < dof - 1; i++) {
    Eigen::Vector3d s = J.col(i);
    q[i] = signed_angle(J_old.col(i + 1), J.col(i + 1), s);
    if (i == dof - 2) break;
    Eigen::Matrix3d R_rot = R_axis_angle(s, q[i]);
    J_old.block(0, i + 1, 3, dof - (i + 1)) = R_rot * J_old.block(0, i + 1, 3, dof - (i + 1));
  }
  return q;
}

// lower rotationa part of J
array<double, 3> q_from_low_J(const Eigen::Matrix<double, 3, 7>& J) {
  const int dof = 4;
  array<double, dof - 1> q;
  Eigen::Matrix<double, 3, 4> J_old_low = J0_S.block(0, 0, 3, dof);  // [3,dof]
  for (int i = 0; i < dof - 1; i++) {
    Eigen::Vector3d s = J.col(i);
    q[i] = signed_angle(J_old_low.col(i + 1), J.col(i + 1), s);
    if (i == dof - 2) break;
    Eigen::Matrix3d R_rot = R_axis_angle(s, q[i]);
    J_old_low.block(0, i + 1, 3, dof - (i + 1)) = R_rot * J_old_low.block(0, i + 1, 3, dof - (i + 1));
  }
  return q;
}

// whole Jacobian and q
array<double, 7> J_to_q(const array<array<double, 6>, 7>& J, const array<array<double, 3>, 3>& R, const char ee) {
  // J is the transpose of the Jacobian
  // R is the rotation matrix of frame ee
  // ee must be a frame attached to the gripper: "E", "F" or "8".
  Eigen::Matrix<double, 3, 7> Jrot;
  Jrot << J[0][0], J[1][0], J[2][0], J[3][0], J[4][0], J[5][0], J[6][0], J[0][1], J[1][1], J[2][1], J[3][1], J[4][1],
      J[5][1], J[6][1], J[0][2], J[1][2], J[2][2], J[3][2], J[4][2], J[5][2], J[6][2];
  const int dof = 7;
  array<double, dof> q;
  array<double, 3> i7, ie, s6, s7;
  s6 = {J[5][0], J[5][1], J[5][2]};
  s7 = {J[6][0], J[6][1], J[6][2]};
  Cross_(s6, s7, i7);
  ie = {R[0][0], R[1][0], R[2][0]};
  q[6] = signed_angle(i7, ie, s7) + (ee == 'E' ? -PI / 4 : 0);
  Eigen::Matrix<double, 3, 7> J_old = J0_S;  // [3,dof]
  for (int i = 0; i < dof - 1; i++) {
    Eigen::Vector3d s = Jrot.col(i);
    q[i] = signed_angle(J_old.col(i + 1), Jrot.col(i + 1), s);
    if (i == dof - 2) break;
    Eigen::Matrix3d R_rot = R_axis_angle(s, q[i]);
    J_old.block(0, i + 1, 3, dof - (i + 1)) = R_rot * J_old.block(0, i + 1, 3, dof - (i + 1));
  }
  return q;
}

void column_1s_times_vec(const array<double, 9>& R, const array<double, 3>& v, array<double, 3>& res) {
  res[0] = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
  res[1] = R[3] * v[0] + R[4] * v[1] + R[5] * v[2];
  res[2] = R[6] * v[0] + R[7] * v[1] + R[8] * v[2];
}

void rotate_by_axis_angle(
    const array<double, 3>& s, const double theta, const array<double, 3>& v, array<double, 3>& res) {
  Eigen::Matrix3d R_rot = R_axis_angle(s, theta);
  res[0] = R_rot(0, 0) * v[0] + R_rot(0, 1) * v[1] + R_rot(0, 2) * v[2];
  res[1] = R_rot(1, 0) * v[0] + R_rot(1, 1) * v[1] + R_rot(1, 2) * v[2];
  res[2] = R_rot(2, 0) * v[0] + R_rot(2, 1) * v[1] + R_rot(2, 2) * v[2];
}

// Rodrigues' formula for the common "rotate one vector" case. Building a full
// 3x3 matrix in thread-local storage and multiplying it by one vector costs far
// more, especially in the sampled swivel solver where this runs hundreds of times.
static Eigen::Vector3d rotate_by_axis_angle(
    const array<double, 3>& axis, const double theta, const Eigen::Vector3d& v) {
  const double ct = cos(theta);
  const double st = sin(theta);
  const double dot = axis[0] * v[0] + axis[1] * v[1] + axis[2] * v[2];
  return Eigen::Vector3d{
      ct * v[0] + st * (axis[1] * v[2] - axis[2] * v[1]) + (1 - ct) * dot * axis[0],
      ct * v[1] + st * (axis[2] * v[0] - axis[0] * v[2]) + (1 - ct) * dot * axis[1],
      ct * v[2] + st * (axis[0] * v[1] - axis[1] * v[0]) + (1 - ct) * dot * axis[2]};
}

Eigen::Matrix4d T_rpy(
    const double r, const double p, const double y, const double px, const double py, const double pz) {
  Eigen::Matrix4d T;
  T << cos(p) * cos(y), cos(y) * sin(p) * sin(r) - cos(r) * sin(y), sin(r) * sin(y) + cos(r) * cos(y) * sin(p), px,
      cos(p) * sin(y), cos(r) * cos(y) + sin(p) * sin(r) * sin(y), cos(r) * sin(p) * sin(y) - cos(y) * sin(r), py,
      -sin(p), cos(p) * sin(r), cos(p) * cos(r), pz, 0, 0, 0, 1;
  return T;
}

Eigen::Matrix4d T_rot_z(const double theta, const double px = 0.0, const double py = 0.0, const double pz = 0.0) {
  Eigen::Matrix4d T;
  T << cos(theta), -sin(theta), 0, px, sin(theta), cos(theta), 0, py, 0, 0, 1, pz, 0, 0, 0, 1;
  return T;
}

Eigen::Matrix<double, 6, 6> Adj_trans(const double x, const double y, const double z) {
  Eigen::Matrix<double, 6, 6> Adj = Eigen::Matrix<double, 6, 6>::Identity();
  Adj.block<3, 3>(3, 0) << 0, -z, y, z, 0, -x, -y, x, 0;
  return Adj;
}

void get_frame_transforms(array<Eigen::Matrix4d, 9>& Ti, const array<double, 7>& q) {
  // gets all the transformation matrices of adjascent frames for joint angles q
  Ti[0] = T_rpy(0, 0, 0, 0, 0, 0.333) * T_rot_z(q[0]);              // T01
  Ti[1] = T_rpy(-PI / 2, 0, 0, 0, 0, 0) * T_rot_z(q[1]);            // T12
  Ti[2] = T_rpy(PI / 2, 0, 0, 0, -0.316, 0) * T_rot_z(q[2]);        // T23
  Ti[3] = T_rpy(PI / 2, 0, 0, 0.0825, 0, 0) * T_rot_z(q[3]);        // T34
  Ti[4] = T_rpy(-PI / 2, 0, 0, -0.0825, 0.384, 0) * T_rot_z(q[4]);  // T45
  Ti[5] = T_rpy(PI / 2, 0, 0, 0, 0, 0) * T_rot_z(q[5]);             // T56
  Ti[6] = T_rpy(PI / 2, 0, 0, 0.088, 0, 0) * T_rot_z(q[6]);         // T67
  Ti[7] = T_rpy(0, 0, 0, 0, 0, 0.107);                              // T78
  Ti[8] = T_rot_z(-PI / 4, 0, 0, 0.1034);                           // T8E
}

unsigned int ee_number(const char ee) {
  if (ee == 'E')  // ask default value first
    return 9;
  if (ee == 'F' || ee == '8') return 8;
  if (ee == '1' || ee == '2' || ee == '3' || ee == '4' || ee == '5' || ee == '6' || ee == '7') return ee - '0';
  return 9;
}

array<array<double, 6>, 7> J_from_q(const array<double, 7>& q, const char ee) {
  // returns J^T for a given vector of joint angles, q. The end-effector frame is ee
  // OUTPUT: J^T \in R^(7,6): array<array<double,6>,7>
  // INPUT: q \in R^7, array<double,7>
  //        ee
  Eigen::Vector3d s, r, m;
  unsigned int een = ee_number(ee);
  unsigned int cols = een >= 7 ? 7 : een;
  Eigen::Matrix<double, 6, 7> J6d;
  array<Eigen::Matrix4d, 9> Ti;
  get_frame_transforms(Ti, q);
  Eigen::Matrix4d T = Ti[0];  // T01
  J6d.col(0) << 0, 0, 1, 0, 0, 0;
  for (unsigned int i = 1; i < een; i++) {
    T = T * Ti[i];  // T0{i+1} = T0{i}*T{i}{i+1}
    if (i < 7) {
      s = T.block<3, 1>(0, 2);
      r = T.block<3, 1>(0, 3);
      m = r.cross(s);
      J6d.col(i) << s[0], s[1], s[2], m[0], m[1], m[2];
    }
  }
  J6d = Adj_trans(-T(0, 3), -T(1, 3), -T(2, 3)) * J6d;
  array<array<double, 6>, 7> Jarr;
  for (unsigned int i = 0; i < cols; i++) Jarr[i] = {J6d(0, i), J6d(1, i), J6d(2, i), J6d(3, i), J6d(4, i), J6d(5, i)};
  for (int i = cols; i < 7; i++) Jarr[i] = {0, 0, 0, 0, 0, 0};
  return Jarr;
}

static void post_rotate_z(Eigen::Matrix3d& rotation, const double angle) {
  const double c = cos(angle);
  const double s = sin(angle);
  const Eigen::Vector3d x = rotation.col(0);
  const Eigen::Vector3d y = rotation.col(1);
  rotation.col(0) = c * x + s * y;
  rotation.col(1) = -s * x + c * y;
}

static void post_rotate_quarter_x_then_z(Eigen::Matrix3d& rotation, const double angle, const int sign) {
  const double c = cos(angle);
  const double s = sin(angle);
  const Eigen::Vector3d x = rotation.col(0);
  const Eigen::Vector3d y = rotation.col(1);
  const Eigen::Vector3d z = rotation.col(2);
  if (sign > 0) {
    rotation.col(0) = c * x + s * z;
    rotation.col(1) = -s * x + c * z;
    rotation.col(2) = -y;
  } else {
    rotation.col(0) = c * x - s * z;
    rotation.col(1) = -s * x - c * z;
    rotation.col(2) = y;
  }
}

static Eigen::Matrix4d franka_flange_fk(const array<double, 7>& q) {
  // The adjacent transforms consist only of z joint rotations and fixed +/-pi/2
  // x rotations. Propagating the three rotation columns directly avoids creating
  // nine 4x4 matrices and performing seven general homogeneous multiplications.
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d position(0.0, 0.0, d1);
  post_rotate_z(rotation, q[0]);
  post_rotate_quarter_x_then_z(rotation, q[1], -1);
  position -= d3 * rotation.col(1);
  post_rotate_quarter_x_then_z(rotation, q[2], 1);
  position += a4 * rotation.col(0);
  post_rotate_quarter_x_then_z(rotation, q[3], 1);
  position += -a5 * rotation.col(0) + d5 * rotation.col(1);
  post_rotate_quarter_x_then_z(rotation, q[4], -1);
  post_rotate_quarter_x_then_z(rotation, q[5], 1);
  position += a7 * rotation.col(0);
  post_rotate_quarter_x_then_z(rotation, q[6], 1);
  position += 0.107 * rotation.col(2);

  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.block<3, 3>(0, 0) = rotation;
  transform.block<3, 1>(0, 3) = position;
  return transform;
}

Eigen::Matrix4d franka_fk(const array<double, 7>& q, const char ee) {
  // Forward kinematics function
  // INPUT: joint angles q, and end effector name ee
  // OUTPUT: TOee is the transformation matrix of frame ee w.r.t. frame O
  if (ee == 'F' || ee == '8') return franka_flange_fk(q);
  if (ee == 'E') {
    Eigen::Matrix4d transform = franka_flange_fk(q);
    Eigen::Matrix3d rotation = transform.block<3, 3>(0, 0);
    transform.block<3, 1>(0, 3) += 0.1034 * rotation.col(2);
    post_rotate_z(rotation, -PI / 4);
    transform.block<3, 3>(0, 0) = rotation;
    return transform;
  }

  unsigned int een = ee_number(ee);
  array<Eigen::Matrix4d, 9> Ti;
  get_frame_transforms(Ti, q);
  Eigen::Matrix4d TOee = Ti[0];
  for (unsigned int i = 1; i < een; i++) TOee = TOee * Ti[i];
  return TOee;
}

void franka_fk_all_frames(array<Eigen::Matrix4d, 9>& Ts, const array<double, 7>& q) {
  // Forward kinematics function saving in Ts the transformation matrices of all frames w.r.t. frame O
  array<Eigen::Matrix4d, 9> Ti;
  get_frame_transforms(Ti, q);
  Ts[0] = Ti[0];
  for (int i = 1; i < 9; i++) Ts[i] = Ts[i - 1] * Ti[i];
}

unsigned int franka_ik_q7(
    const array<double, 3>& r, const array<double, 9>& ROE, const double q7, array<array<double, 7>, 8>& qsols,
    const double q1_sing) {
  // IK with q7 as free variable
  // INPUT: r = r_EO_O, position of frame E in frame O
  //        ROE, orientation of frame E in frame O
  //        q7, joint angle of joint 7
  //        qsols, array to store 8 solutions
  //        q1_sing, emergency value of q1 in case of singularity at shoulder joints.
  // OUTPUT: number of solutions found.
  // ri = r_iS_O, i = 1,2,3,4,5,6,7
  // si = s_i_O
  Eigen::Vector3d i_E_O(ROE[0], ROE[3], ROE[6]);
  array<double, 3> k_E_O = {ROE[2], ROE[5], ROE[8]};
  Eigen::Vector3d i_6_O = rotate_by_axis_angle(k_E_O, -(q7 - PI / 4), i_E_O);
  array<double, 3> s6;
  Cross_(k_E_O, i_6_O, s6);
  array<double, 3> r6 = {
      r[0] - dE * k_E_O[0] - a7 * i_6_O[0],
      r[1] - dE * k_E_O[1] - a7 * i_6_O[1],
      r[2] - d1 - dE * k_E_O[2] - a7 * i_6_O[2]};
  double l = Norm(r6);
  double tmp = (b1 * b1 - l * l - b2 * b2) / (-2 * l * b2);
  if (tmp > 1) {
    if ((tmp - 1) * (tmp - 1) < SING_TOL * SING_TOL) {
      tmp = 1;
    } else {
      geofik_log() << "ERROR: unable to assembly kinematic chain";
      for (int i = 0; i < 8; ++i) {
        fill(qsols[i].begin(), qsols[i].end(), NAN);
      }
      return 0;
    }
  }
  double actmp = acos(tmp);
  double alpha2 = beta2 + actmp;
  array<double, 3> k_C_O = {-r6[0] / l, -r6[1] / l, -r6[2] / l};
  array<double, 3> i_C_O;
  Cross_(k_C_O, s6, i_C_O);
  tmp = Norm(i_C_O);
  i_C_O = {i_C_O[0] / tmp, i_C_O[1] / tmp, i_C_O[2] / tmp};
  array<double, 3> j_C_O;
  Cross_(k_C_O, i_C_O, j_C_O);
  double ry = s6[0] * j_C_O[0] + s6[1] * j_C_O[1] + s6[2] * j_C_O[2];
  double rz = s6[0] * k_C_O[0] + s6[1] * k_C_O[1] + s6[2] * k_C_O[2];
  array<array<double, 3>, 4> s5s;
  double sa2, ca2;
  int n_alphs = 1;
  unsigned int n_sols = 0;
  if (d3 + d5 < l && l < b1 + b2) n_alphs = 2;
  double v[3];
  for (int i = 0; i < n_alphs; i++) {
    // franky vendoring patch: upstream advanced alpha2 to the second elbow assembly at
    // the *end* of the loop body, where the `continue` below skips past it -- so a first
    // branch that bailed left the second iteration re-solving the identical branch
    // instead of the elbow flip. Selecting from the index makes the two iterations
    // independent. See README.md.
    alpha2 = (i == 0) ? (beta2 + actmp) : (beta2 - actmp);
    sa2 = sin(alpha2);
    ca2 = cos(alpha2);
    tmp = -rz * ca2 / (ry * sa2);
    if (tmp * tmp > 1) {
      if (fabs(fabs(tmp) - 1.0) < TRIG_DOMAIN_TOL)
        tmp = tmp > 0.0 ? 1.0 : -1.0;
      else
        continue;
    }
    tmp = asin(tmp);
    v[0] = -sa2 * cos(tmp);
    v[1] = -sa2 * sin(tmp);
    v[2] = -ca2;
    s5s[n_sols] = {
        i_C_O[0] * v[0] + j_C_O[0] * v[1] + k_C_O[0] * v[2],
        i_C_O[1] * v[0] + j_C_O[1] * v[1] + k_C_O[1] * v[2],
        i_C_O[2] * v[0] + j_C_O[2] * v[1] + k_C_O[2] * v[2]};
    tmp = 2 * sa2 * cos(tmp);
    // s5[n_sols+1] = s5s[n_sols] + (2*sa2*cos(tmp)*i_C_O);
    s5s[n_sols + 1] = {
        s5s[n_sols][0] + tmp * i_C_O[0], s5s[n_sols][1] + tmp * i_C_O[1], s5s[n_sols][2] + tmp * i_C_O[2]};
    n_sols += 2;
  }
  array<double, 6> sol1;
  array<double, 3> sol2;
  array<double, 3> s4, r4, s3, s2, s5;
  for (unsigned int i = 0; i < n_sols; i++) {
    s5 = s5s[i];
    Cross_(s5, r6, s4);
    tmp = Norm(s4);
    s4 = {s4[0] / tmp, s4[1] / tmp, s4[2] / tmp};
    Cross_(s5, s4, r4);
    r4 = {r6[0] - d5 * s5[0] + a5 * r4[0], r6[1] - d5 * s5[1] + a5 * r4[1], r6[2] - d5 * s5[2] + a5 * r4[2]};
    Eigen::Matrix3d R_tmp = R_axis_angle(s4, beta1);
    s3 = {
        R_tmp(0, 0) * r4[0] + R_tmp(0, 1) * r4[1] + R_tmp(0, 2) * r4[2],
        R_tmp(1, 0) * r4[0] + R_tmp(1, 1) * r4[1] + R_tmp(1, 2) * r4[2],
        R_tmp(2, 0) * r4[0] + R_tmp(2, 1) * r4[1] + R_tmp(2, 2) * r4[2]};
    tmp = Norm(s3);
    s3 = {s3[0] / tmp, s3[1] / tmp, s3[2] / tmp};
    tmp = s3[1] * s3[1] + s3[0] * s3[0];
    if (tmp > SHOULDER_SING_TOL * SHOULDER_SING_TOL) {
      s2 = {-s3[1] / sqrt(tmp), s3[0] / sqrt(tmp), 0};
    } else {
      s2 = {sin(q1_sing), cos(q1_sing), 0};
    }
    Eigen::Matrix<double, 3, 7> tmp_J = J_dir(s2, s3, s4, s5, s6, k_E_O);
    sol1 = q_from_J(tmp_J);
    tmp_J.col(1) = -1 * tmp_J.col(1);
    sol2 = q_from_low_J(tmp_J);
    qsols[2 * i] = {sol1[0], sol1[1], sol1[2], sol1[3], sol1[4], sol1[5], q7};
    check_limits(qsols[2 * i], 7);
    qsols[2 * i + 1] = {sol2[0], sol2[1], sol2[2], qsols[2 * i][3], qsols[2 * i][4], qsols[2 * i][5], qsols[2 * i][6]};
    check_limits(qsols[2 * i + 1], 3);
  }
  for (int i = 2 * n_sols; i < 8; i++) fill(qsols[i].begin(), qsols[i].end(), NAN);
  return 2 * n_sols;
}

unsigned int franka_ik_q4(
    const array<double, 3>& r, const array<double, 9>& ROE, const double q4, array<array<double, 7>, 8>& qsols,
    const double q1_sing, const double q7_sing) {
  // IK with q4 as free variable
  // INPUT: r = r_EO_O, position of frame E in frame O
  //        ROE, orientation of frame E in frame O (row-first format)
  //        q4, joint angle of joint 4
  //        qsols, array to store 8 solutions
  //        q1_sing, emergency value of q1 in case of singularity at shoulder joints (type-1 singularity).
  //        q7_sing, emergency value of q7 in case of singularity of S7 intersecting S (type-2 singularity).
  // OUTPUT: number of solutions found.
  // ri = r_iS_O, i = 1,2,3,4,5,6,7
  // si = s_i_O
  array<double, 3> r_ES_O = {r[0], r[1], r[2] - d1};
  array<double, 3> tmp_v = {
      r_ES_O[1] * ROE[8] - r_ES_O[2] * ROE[5],
      r_ES_O[2] * ROE[2] - r_ES_O[0] * ROE[8],
      r_ES_O[0] * ROE[5] - r_ES_O[1] * ROE[2]};
  if (tmp_v[0] * tmp_v[0] + tmp_v[1] * tmp_v[1] + tmp_v[2] * tmp_v[2] < SING_TOL * SING_TOL)
    return franka_ik_q7(r, ROE, q7_sing, qsols, q1_sing);
  array<double, 3> r_O7S_O = {r_ES_O[0] - dE * ROE[2], r_ES_O[1] - dE * ROE[5], r_ES_O[2] - dE * ROE[8]};
  array<double, 3> r_O7S_E = {
      ROE[0] * r_O7S_O[0] + ROE[3] * r_O7S_O[1] + ROE[6] * r_O7S_O[2],
      ROE[1] * r_O7S_O[0] + ROE[4] * r_O7S_O[1] + ROE[7] * r_O7S_O[2],
      ROE[2] * r_O7S_O[0] + ROE[5] * r_O7S_O[1] + ROE[8] * r_O7S_O[2]};
  double alpha = q4 + beta1 + beta2 - PI;
  double lo2 = b1 * b1 + b2 * b2 - 2 * b1 * b2 * cos(alpha);
  double lp2 = lo2 - r_O7S_E[2] * r_O7S_E[2];
  // fixed triangle condition
  if (lp2 < 0) {
    if (lp2 * lp2 < SING_TOL * SING_TOL)
      lp2 = 0;
    else {
      geofik_log() << "\nERROR: unable to assembly kinematic chain\n";
      for (int i = 0; i < 8; ++i) fill(qsols[i].begin(), qsols[i].end(), NAN);
      return 0;
    }
  }
  double gamma2 = beta2 + asin(b1 * sin(alpha) / sqrt(lo2));
  double cg2 = cos(gamma2), sg2 = sin(gamma2);
  double Lp2 = r_O7S_E[0] * r_O7S_E[0] + r_O7S_E[1] * r_O7S_E[1], phi = atan2(-r_O7S_E[1], -r_O7S_E[0]);
  double tmp = (Lp2 + a7 * a7 - lp2) / (2 * sqrt(Lp2) * a7);
  // fixed condition
  if (tmp > 1.0) {
    if ((tmp - 1) * (tmp - 1) < SING_TOL * SING_TOL)
      tmp = 1.0;
    else {
      geofik_log() << "\nERROR: unable to assembly kinematic chain\n";
      for (int i = 0; i < 8; ++i) {
        fill(qsols[i].begin(), qsols[i].end(), NAN);
      }
      return 0;
    }
  }
  /*
  if ((tmp - 1) * (tmp - 1) < SING_TOL*SING_TOL)
      tmp = 1.0;
  if (tmp > 1.0) {
      geofik_log() << "\nERROR: unable to assembly kinematic chain\n";
      for (int i = 0; i < 8; ++i) {
          fill(qsols[i].begin(), qsols[i].end(), NAN);
      }
      return 0;
  }
  */
  double psi = acos(tmp), ry, rz;
  double q7s[2] = {-phi - psi - 3 * PI / 4, -phi + psi - 3 * PI / 4};
  double gammas[2] = {0, 0};
  unsigned int ind = 0;
  array<double, 3> s2, s3, s4, s5, s6, r4, r6, i_C_O, j_C_O, k_C_O;
  array<double, 6> sol1;
  array<double, 3> sol2;
  for (auto q7 : q7s) {
    tmp_v = {cos(-q7 + 3 * PI / 4), sin(-q7 + 3 * PI / 4), 0};
    s6 = {
        ROE[0] * tmp_v[0] + ROE[1] * tmp_v[1],
        ROE[3] * tmp_v[0] + ROE[4] * tmp_v[1],
        ROE[6] * tmp_v[0] + ROE[7] * tmp_v[1]};
    tmp_v = {-a7 * cos(-q7 + PI / 4), -a7 * sin(-q7 + PI / 4), 0};
    r6 = {
        ROE[0] * tmp_v[0] + ROE[1] * tmp_v[1],
        ROE[3] * tmp_v[0] + ROE[4] * tmp_v[1],
        ROE[6] * tmp_v[0] + ROE[7] * tmp_v[1]};
    r6 = {r6[0] + r_O7S_O[0], r6[1] + r_O7S_O[1], r6[2] + r_O7S_O[2]};
    tmp = Norm(r6);
    k_C_O = {-r6[0] / tmp, -r6[1] / tmp, -r6[2] / tmp};
    Cross_(k_C_O, s6, i_C_O);
    tmp = Norm(i_C_O);
    i_C_O = {i_C_O[0] / tmp, i_C_O[1] / tmp, i_C_O[2] / tmp};
    Cross_(k_C_O, i_C_O, j_C_O);
    ry = s6[0] * j_C_O[0] + s6[1] * j_C_O[1] + s6[2] * j_C_O[2];
    rz = s6[0] * k_C_O[0] + s6[1] * k_C_O[1] + s6[2] * k_C_O[2];
    tmp = -rz * cg2 / (ry * sg2);
    if (tmp * tmp > 1) continue;
    tmp = asin(tmp);
    gammas[0] = tmp;
    gammas[1] = PI - tmp;
    for (auto gamma : gammas) {
      tmp_v = {-sg2 * cos(gamma), -sg2 * sin(gamma), -cg2};
      s5 = {
          i_C_O[0] * tmp_v[0] + j_C_O[0] * tmp_v[1] + k_C_O[0] * tmp_v[2],
          i_C_O[1] * tmp_v[0] + j_C_O[1] * tmp_v[1] + k_C_O[1] * tmp_v[2],
          i_C_O[2] * tmp_v[0] + j_C_O[2] * tmp_v[1] + k_C_O[2] * tmp_v[2]};
      Cross_(s5, r6, s4);
      tmp = Norm(s4);
      s4 = {s4[0] / tmp, s4[1] / tmp, s4[2] / tmp};
      Cross_(s5, s4, r4);
      r4 = {r6[0] - d5 * s5[0] + a5 * r4[0], r6[1] - d5 * s5[1] + a5 * r4[1], r6[2] - d5 * s5[2] + a5 * r4[2]};
      Eigen::Matrix3d R_tmp = R_axis_angle(s4, beta1);
      s3 = {
          R_tmp(0, 0) * r4[0] + R_tmp(0, 1) * r4[1] + R_tmp(0, 2) * r4[2],
          R_tmp(1, 0) * r4[0] + R_tmp(1, 1) * r4[1] + R_tmp(1, 2) * r4[2],
          R_tmp(2, 0) * r4[0] + R_tmp(2, 1) * r4[1] + R_tmp(2, 2) * r4[2]};
      tmp = Norm(s3);
      s3 = {s3[0] / tmp, s3[1] / tmp, s3[2] / tmp};
      tmp = s3[1] * s3[1] + s3[0] * s3[0];
      if (tmp > SHOULDER_SING_TOL * SHOULDER_SING_TOL)
        s2 = {-s3[1] / sqrt(tmp), s3[0] / sqrt(tmp), 0};
      else
        s2 = {sin(q1_sing), cos(q1_sing), 0};
      Eigen::Matrix<double, 3, 7> tmp_J = J_dir(s2, s3, s4, s5, s6, array<double, 3>{ROE[2], ROE[5], ROE[8]});
      sol1 = q_from_J(tmp_J);
      tmp_J.col(1) = -1 * tmp_J.col(1);
      sol2 = q_from_low_J(tmp_J);
      qsols[2 * ind] = {sol1[0], sol1[1], sol1[2], sol1[3], sol1[4], sol1[5], q7};
      check_limits(qsols[2 * ind], 7);
      qsols[2 * ind + 1] = {
          sol2[0], sol2[1], sol2[2], qsols[2 * ind][3], qsols[2 * ind][4], qsols[2 * ind][5], qsols[2 * ind][6]};
      check_limits(qsols[2 * ind + 1], 3);
      ind++;
    }
  }
  for (int i = 2 * ind; i < 8; ++i) {
    fill(qsols[i].begin(), qsols[i].end(), NAN);
  }
  return 2 * ind;
}

unsigned int franka_ik_q6_parallel(
    const array<double, 3>& r_ES_O, const array<double, 9>& ROE, const int sgn, array<array<double, 7>, 8>& qsols,
    const double q1_sing) {
  // Parallel case of the IK with q6 as free variable. Only called by franka_ik_q6(), not by the user.
  // INPUT: r_ES_O, ROE, sgn  = sign(cos(q6)), qsols, q1_sing.
  // OUTPUT: number of solutions found.
  // NOTATION:
  // ri = r_iS_O, i = 1,2,3,4,5,6,7
  // si = s_i_O
  // Q is a frame that is parallel to frame E and has origin at Q
  array<double, 3> s7 = {ROE[2], ROE[5], ROE[8]};
  array<double, 3> r_QS_O = {
      r_ES_O[0] + (-dE + sgn * d5) * s7[0], r_ES_O[1] + (-dE + sgn * d5) * s7[1], r_ES_O[2] + (-dE + sgn * d5) * s7[2]};
  array<double, 3> r_SQ_Q = {
      -ROE[0] * r_QS_O[0] - ROE[3] * r_QS_O[1] - ROE[6] * r_QS_O[2],
      -ROE[1] * r_QS_O[0] - ROE[4] * r_QS_O[1] - ROE[7] * r_QS_O[2],
      -ROE[2] * r_QS_O[0] - ROE[5] * r_QS_O[1] - ROE[8] * r_QS_O[2]};
  double tmp = b1 * b1 - r_SQ_Q[2] * r_SQ_Q[2];
  if (tmp * tmp < SING_TOL * SING_TOL) tmp = 0;
  if (tmp < 0) {
    geofik_log() << "ERROR: unable to assembly kinematic chain";
    for (int i = 0; i < 8; ++i) {
      fill(qsols[i].begin(), qsols[i].end(), NAN);
    }
    return 0;
  }
  double lp = sqrt(tmp);
  array<double, 3> r_SpQ_Q = {r_SQ_Q[0], r_SQ_Q[1], 0};
  double l_SpQ = sqrt(r_SQ_Q[0] * r_SQ_Q[0] + r_SQ_Q[1] * r_SQ_Q[1]);
  double alphas[2], Ls[2];
  double q7;
  Ls[0] = a5 + lp;
  Ls[1] = a5 - lp;
  array<double, 3> tmp_v, r_O6pQ_Q, i_4_Q, r_O4Q_Q, s6_Q, r_O6_Q, s4_Q, s3_Q, s2, s3, s4, s5, s6;
  Eigen::Matrix<double, 3, 4> partial_J_Q, partial_J_O;
  Eigen::Matrix3d ROQ;
  ROQ << ROE[0], ROE[1], ROE[2], ROE[3], ROE[4], ROE[5], ROE[6], ROE[7], ROE[8];
  const array<double, 3> k{{0, 0, 1}};
  array<double, 3> s5_Q{{0, 0, -1.0 * sgn}};
  int tmp_sgn;
  unsigned int ind = 0;
  array<double, 6> sol1;
  array<double, 3> sol2;
  for (auto L : Ls) {
    tmp = (-L * L + a7 * a7 + l_SpQ * l_SpQ) / (2 * a7 * l_SpQ);
    if ((tmp - 1) * (tmp - 1) < SING_TOL * SING_TOL)
      tmp = 1;
    else if ((tmp + 1) * (tmp + 1) < SING_TOL * SING_TOL)
      tmp = -1;
    if (tmp * tmp > 1) continue;
    alphas[0] = acos(tmp);
    alphas[1] = -acos(tmp);
    for (auto alpha : alphas) {
      rotate_by_axis_angle(k, alpha, r_SpQ_Q, r_O6pQ_Q);
      r_O6pQ_Q = {a7 * r_O6pQ_Q[0] / l_SpQ, a7 * r_O6pQ_Q[1] / l_SpQ, a7 * r_O6pQ_Q[2] / l_SpQ};
      i_4_Q = {r_SpQ_Q[0] - r_O6pQ_Q[0], r_SpQ_Q[1] - r_O6pQ_Q[1], r_SpQ_Q[2] - r_O6pQ_Q[2]};
      tmp = Norm(i_4_Q);
      tmp_sgn = L < 0 ? -1 : 1;
      i_4_Q = {tmp_sgn * i_4_Q[0] / tmp, tmp_sgn * i_4_Q[1] / tmp, tmp_sgn * i_4_Q[2] / tmp};
      r_O4Q_Q = {r_O6pQ_Q[0] + a5 * i_4_Q[0], r_O6pQ_Q[1] + a5 * i_4_Q[1], r_O6pQ_Q[2] + a5 * i_4_Q[2]};
      Cross_(r_O6pQ_Q, k, s6_Q);
      r_O6_Q = {r_O6pQ_Q[0], r_O6pQ_Q[1], r_O6pQ_Q[2] - sgn * d5};
      Cross_(i_4_Q, s5_Q, s4_Q);
      tmp_v = {r_O4Q_Q[0] - r_SQ_Q[0], r_O4Q_Q[1] - r_SQ_Q[1], r_O4Q_Q[2] - r_SQ_Q[2]};
      rotate_by_axis_angle(s4_Q, beta1, tmp_v, s3_Q);
      tmp = Norm(s3_Q);
      partial_J_Q << s3_Q[0] / tmp, s4_Q[0], s5_Q[0], s6_Q[0], s3_Q[1] / tmp, s4_Q[1], s5_Q[1], s6_Q[1], s3_Q[2] / tmp,
          s4_Q[2], s5_Q[2], s6_Q[2];
      partial_J_O = ROQ * partial_J_Q;
      s3 = {partial_J_O(0, 0), partial_J_O(1, 0), partial_J_O(2, 0)};
      s4 = {partial_J_O(0, 1), partial_J_O(1, 1), partial_J_O(2, 1)};
      s5 = {partial_J_O(0, 2), partial_J_O(1, 2), partial_J_O(2, 2)};
      s6 = {partial_J_O(0, 3), partial_J_O(1, 3), partial_J_O(2, 3)};
      q7 = atan2(r_O6pQ_Q[1], -r_O6pQ_Q[0]) + PI / 4;
      tmp = s3[1] * s3[1] + s3[0] * s3[0];
      if (tmp > SHOULDER_SING_TOL * SHOULDER_SING_TOL)
        s2 = {-s3[1] / sqrt(tmp), s3[0] / sqrt(tmp), 0};
      else
        s2 = {sin(q1_sing), cos(q1_sing), 0};
      Eigen::Matrix<double, 3, 7> tmp_J = J_dir(s2, s3, s4, s5, s6, s7);
      sol1 = q_from_J(tmp_J);
      tmp_J.col(1) = -1 * tmp_J.col(1);
      sol2 = q_from_low_J(tmp_J);
      qsols[2 * ind] = {sol1[0], sol1[1], sol1[2], sol1[3], sol1[4], sol1[5], q7};
      check_limits(qsols[2 * ind], 7);
      qsols[2 * ind + 1] = {
          sol2[0], sol2[1], sol2[2], qsols[2 * ind][3], qsols[2 * ind][4], qsols[2 * ind][5], qsols[2 * ind][6]};
      check_limits(qsols[2 * ind + 1], 3);
      ind++;
    }
  }
  for (int i = 2 * ind; i < 8; ++i) {
    fill(qsols[i].begin(), qsols[i].end(), NAN);
  }
  return 2 * ind;
}

unsigned int franka_ik_q6(
    const array<double, 3>& r, const array<double, 9>& ROE, const double q6, array<array<double, 7>, 8>& qsols,
    const double q1_sing, const double q7_sing) {
  // IK with q6 as free variable
  // INPUT: r = r_EO_O, position of frame E in frame O
  //        ROE, orientation of frame E in frame O (row-first format)
  //        q6, joint angle of joint 6
  //        qsols, array to store 8 solutions
  //        q1_sing, emergency value of q1 in case of singularity at shoulder joints (type-1 singularity).
  //        q7_sing, emergency value of q7 in case of singularity of S7 intersecting S (type-2 singularity).
  // OUTPUT: number of solutions found.
  // NOTATION:
  // ri = r_iS_O, i = 1,2,3,4,5,6,7
  // si = s_i_O
  array<double, 3> r_ES_O = {r[0], r[1], r[2] - d1};
  array<double, 3> tmp_v = {
      r_ES_O[1] * ROE[8] - r_ES_O[2] * ROE[5],
      r_ES_O[2] * ROE[2] - r_ES_O[0] * ROE[8],
      r_ES_O[0] * ROE[5] - r_ES_O[1] * ROE[2]};
  if (tmp_v[0] * tmp_v[0] + tmp_v[1] * tmp_v[1] + tmp_v[2] * tmp_v[2] < SING_TOL * SING_TOL)
    return franka_ik_q7(r, ROE, q7_sing, qsols, q1_sing);
  if (sin(q6) * sin(q6) < WRIST_PARALLEL_TOL * WRIST_PARALLEL_TOL)  // PARALLEL CASE:
    return franka_ik_q6_parallel(r_ES_O, ROE, cos(q6) >= 0 ? 1 : -1, qsols, q1_sing);

  // NON-PARALLEL CASE:
  array<double, 3> s7 = {ROE[2], ROE[5], ROE[8]};
  double gamma1 = PI - q6;
  double cg1 = cos(gamma1);
  double sg1 = sin(gamma1);
  array<double, 3> r_O7S_O = {r_ES_O[0] - dE * ROE[2], r_ES_O[1] - dE * ROE[5], r_ES_O[2] - dE * ROE[8]};
  array<double, 3> r_PS_O = {
      r_O7S_O[0] + (a7 / tan(gamma1)) * s7[0],
      r_O7S_O[1] + (a7 / tan(gamma1)) * s7[1],
      r_O7S_O[2] + (a7 / tan(gamma1)) * s7[2]};
  double lP = Norm(r_PS_O);
  double lC = a7 / sg1;
  double Cx = -(ROE[0] * r_PS_O[0] + ROE[3] * r_PS_O[1] + ROE[6] * r_PS_O[2]);
  double Cy = -(ROE[1] * r_PS_O[0] + ROE[4] * r_PS_O[1] + ROE[7] * r_PS_O[2]);
  double Cz = -(ROE[2] * r_PS_O[0] + ROE[5] * r_PS_O[1] + ROE[8] * r_PS_O[2]);
  double c = sqrt(a5 * a5 + (lC + d5) * (lC + d5));
  double tmp = (-b1 * b1 + lP * lP + c * c) / (2 * lP * c);
  // cout << "tmp at triangle: " << tmp;
  if (tmp > 1.0) {
    if ((tmp - 1) * (tmp - 1) < SING_TOL * SING_TOL)
      tmp = 1.0;
    else {
      geofik_log() << "ERROR: unable to assembly kinematic chain";
      for (int i = 0; i < 8; ++i) {
        fill(qsols[i].begin(), qsols[i].end(), NAN);
      }
      return 0;
    }
  }
  double tau = acos(tmp);
  unsigned int n_gamma_sols = 1;
  if ((d3 + d5 + lC < lP) && (lP < b1 + c)) n_gamma_sols = 2;
  double gamma2s[2];
  if (d5 < -lC)
    gamma2s[0] = tau + atan(a5 / (d5 + lC)) + PI;
  else
    gamma2s[0] = tau + atan(a5 / (d5 + lC));
  if (n_gamma_sols > 1) gamma2s[1] = gamma2s[0] - 2 * tau;

  array<array<double, 3>, 4> s5s;
  double q7s[4];
  double d, u1, u2;
  unsigned int n_sols = 0;
  for (unsigned int i = 0; i < n_gamma_sols; i++) {
    d = lP * cos(gamma2s[i]);
    tmp = (d + Cz * cg1) / (sqrt(Cx * Cx * sg1 * sg1 + Cy * Cy * sg1 * sg1));
    // cout << "tmp at cone: " << tmp;
    if (tmp > 1) {
      if ((tmp - 1) * (tmp - 1) < TRIG_DOMAIN_TOL * TRIG_DOMAIN_TOL)
        tmp = 1;
      else
        continue;
    } else if (tmp < -1) {
      if ((tmp + 1) * (tmp + 1) < TRIG_DOMAIN_TOL * TRIG_DOMAIN_TOL)
        tmp = -1;
      else
        continue;
    }
    u1 = asin(tmp);
    u2 = atan2(Cx * sg1, Cy * sg1);
    // cout << "u1 = " << u1 << endl;
    // cout << "u2 = " << u2 << endl;
    q7s[n_sols] = 5 * PI / 4 - u1 + u2;
    tmp_v = {-sg1 * cos(u1 - u2), -sg1 * sin(u1 - u2), cg1};
    column_1s_times_vec(ROE, tmp_v, s5s[n_sols]);
    n_sols++;
    q7s[n_sols] = PI / 4 + u1 + u2;
    tmp_v = {-sg1 * cos(PI - u1 - u2), -sg1 * sin(PI - u1 - u2), cg1};
    column_1s_times_vec(ROE, tmp_v, s5s[n_sols]);
    n_sols++;
  }

  array<double, 3> s2, s3, s4, s6, r4, r6;
  array<double, 6> sol1;
  array<double, 3> sol2;
  unsigned int assembled_sols = 0;
  for (unsigned int i = 0; i < n_sols; i++) {
    r6 = {r_PS_O[0] - lC * s5s[i][0], r_PS_O[1] - lC * s5s[i][1], r_PS_O[2] - lC * s5s[i][2]};
    tmp_v = {r_O7S_O[0] - r6[0], r_O7S_O[1] - r6[1], r_O7S_O[2] - r6[2]};
    Cross_(s7, tmp_v, s6);
    tmp = Norm(s6);
    s6 = {s6[0] / tmp, s6[1] / tmp, s6[2] / tmp};
    Cross_(s5s[i], r6, s4);
    tmp = Norm(s4);
    s4 = {s4[0] / tmp, s4[1] / tmp, s4[2] / tmp};
    Cross_(s5s[i], s4, tmp_v);
    r4 = {
        r6[0] - d5 * s5s[i][0] + a5 * tmp_v[0],
        r6[1] - d5 * s5s[i][1] + a5 * tmp_v[1],
        r6[2] - d5 * s5s[i][2] + a5 * tmp_v[2]};
    tmp = Norm(r4);
    // r4 is the shoulder-to-joint-4 vector and must have the fixed upper-arm
    // length. Cone roots that violate this invariant are algebraic artifacts.
    if (fabs(tmp - b1) > SING_TOL) continue;
    rotate_by_axis_angle(s4, beta1, r4, s3);
    s3 = {s3[0] / tmp, s3[1] / tmp, s3[2] / tmp};
    tmp = s3[1] * s3[1] + s3[0] * s3[0];
    if (tmp > SHOULDER_SING_TOL * SHOULDER_SING_TOL)
      s2 = {-s3[1] / sqrt(tmp), s3[0] / sqrt(tmp), 0};
    else
      s2 = {sin(q1_sing), cos(q1_sing), 0};
    Eigen::Matrix<double, 3, 7> tmp_J = J_dir(s2, s3, s4, s5s[i], s6, s7);
    sol1 = q_from_J(tmp_J);
    tmp_J.col(1) = -1 * tmp_J.col(1);
    sol2 = q_from_low_J(tmp_J);
    qsols[2 * assembled_sols] = {sol1[0], sol1[1], sol1[2], sol1[3], sol1[4], sol1[5], q7s[i]};
    check_limits(qsols[2 * assembled_sols], 7);
    qsols[2 * assembled_sols + 1] = {
        sol2[0],
        sol2[1],
        sol2[2],
        qsols[2 * assembled_sols][3],
        qsols[2 * assembled_sols][4],
        qsols[2 * assembled_sols][5],
        qsols[2 * assembled_sols][6]};
    check_limits(qsols[2 * assembled_sols + 1], 3);
    assembled_sols++;
  }
  for (int i = 2 * assembled_sols; i < 8; ++i) {
    fill(qsols[i].begin(), qsols[i].end(), NAN);
  }
  return 2 * assembled_sols;
}

// FUNCTIONS FOR SWIVEL ANGLE

struct SwivelSample {
  array<double, 2> signed_errors;
  array<double, 2> q7s;
};

static SwivelSample theta_err_from_q7_explore_around_sing(
    const double q7m, const double theta, const Eigen::Vector3d& i_E_O, const array<double, 3>& k_E_O,
    Eigen::Vector3d& i_6_O, const array<double, 3>& n1_O, const array<double, 3>& r_O7S_O,
    const array<double, 3>& u_O7S_O, const double q7step, const unsigned int n_fine) {
  array<double, 3> s6, r6, k_C_O, i_C_O, j_C_O, s4, r4, n2_O, v;
  array<double, 2> errs;
  array<double, 2> signed_errs;
  array<array<double, 3>, 2> s5s;
  double l, tmp, actmp, alpha2, ry, rz, sa2, ca2;
  array<double, 2> best_q7s = {q7m, q7m};
  array<double, 2> min_errs = {1e15, 1e15};
  array<double, 2> min_signed_errs = {1e15, 1e15};
  double q7up = q7m + q7step / 2;
  double q7j = q7m - q7step / 2;
  while (q7j < q7up) {
    i_6_O = rotate_by_axis_angle(k_E_O, -(q7j - PI / 4), i_E_O);
    Cross_(k_E_O, i_6_O, s6);
    r6 = {r_O7S_O[0] - a7 * i_6_O[0], r_O7S_O[1] - a7 * i_6_O[1], r_O7S_O[2] - a7 * i_6_O[2]};
    l = Norm(r6);
    tmp = (b1 * b1 - l * l - b2 * b2) / (-2 * l * b2);
    if (tmp > 1) {
      if ((tmp - 1) * (tmp - 1) < SING_TOL * SING_TOL)
        tmp = 1;
      else {
        q7j += q7step / (2 * n_fine);
        continue;
      }
    }
    actmp = acos(tmp);
    alpha2 = beta2 + actmp;
    k_C_O = {-r6[0] / l, -r6[1] / l, -r6[2] / l};
    i_C_O = Cross(k_C_O, s6);
    tmp = Norm(i_C_O);
    i_C_O = {i_C_O[0] / tmp, i_C_O[1] / tmp, i_C_O[2] / tmp};
    Cross_(k_C_O, i_C_O, j_C_O);
    ry = s6[0] * j_C_O[0] + s6[1] * j_C_O[1] + s6[2] * j_C_O[2];
    rz = s6[0] * k_C_O[0] + s6[1] * k_C_O[1] + s6[2] * k_C_O[2];
    sa2 = sin(alpha2);
    ca2 = cos(alpha2);
    tmp = -rz * ca2 / (ry * sa2);
    if (tmp * tmp > 1) {
      q7j += q7step / (2 * n_fine);
      continue;
    }
    tmp = asin(tmp);
    v = {-sa2 * cos(tmp), -sa2 * sin(tmp), -ca2};
    s5s[0] = {
        i_C_O[0] * v[0] + j_C_O[0] * v[1] + k_C_O[0] * v[2],
        i_C_O[1] * v[0] + j_C_O[1] * v[1] + k_C_O[1] * v[2],
        i_C_O[2] * v[0] + j_C_O[2] * v[1] + k_C_O[2] * v[2]};
    tmp = 2 * sa2 * cos(tmp);
    s5s[1] = {s5s[0][0] + tmp * i_C_O[0], s5s[0][1] + tmp * i_C_O[1], s5s[0][2] + tmp * i_C_O[2]};
    for (int i = 0; i < 2; i++) {
      s4 = Cross(s5s[i], r6);
      tmp = Norm(s4);
      s4 = {s4[0] / tmp, s4[1] / tmp, s4[2] / tmp};
      Cross_(s5s[i], s4, r4);
      r4 = {
          r6[0] - d5 * s5s[i][0] + a5 * r4[0],
          r6[1] - d5 * s5s[i][1] + a5 * r4[1],
          r6[2] - d5 * s5s[i][2] + a5 * r4[2]};
      Cross_(r_O7S_O, r4, n2_O);
      tmp = Dot(n2_O, s4);
      if (tmp < 0) n2_O = {-n2_O[0], -n2_O[1], -n2_O[2]};
      // franky vendoring patch: no longer squared, see theta_err_from_q7.
      signed_errs[i] = wrapped_angle_difference(theta, signed_angle(n1_O, n2_O, u_O7S_O));
      errs[i] = fabs(signed_errs[i]);
    }
    for (int i = 0; i < 2; ++i) {
      if (errs[i] < min_errs[i]) {
        best_q7s[i] = q7j;
        min_errs[i] = errs[i];
        min_signed_errs[i] = signed_errs[i];
      }
    }
    q7j += q7step / (2 * n_fine);
  }
  return SwivelSample{min_signed_errs, best_q7s};
}

static SwivelSample theta_err_from_q7(
    const double q7, const double theta, const Eigen::Vector3d& i_E_O, const array<double, 3>& k_E_O,
    Eigen::Vector3d& i_6_O, const array<double, 3>& n1_O, const array<double, 3>& r_O7S_O,
    const array<double, 3>& u_O7S_O, const double q7step, const unsigned int n_fine) {
  // Calculates the error in swivel angle given the necessary geometry, q7, and the desired swivel angle theta
  // NOTATION: u_O7S_O = r_O7S_O/norm(r_O7S_O), precalculated to improve speed
  array<double, 2> signed_errs;
  i_6_O = rotate_by_axis_angle(k_E_O, -(q7 - PI / 4), i_E_O);
  array<double, 3> s6;
  Cross_(k_E_O, i_6_O, s6);
  // r6 = r_O7S_O - a7 * i_6_O
  array<double, 3> r6 = {r_O7S_O[0] - a7 * i_6_O[0], r_O7S_O[1] - a7 * i_6_O[1], r_O7S_O[2] - a7 * i_6_O[2]};
  double l = Norm(r6);
  double tmp = (b1 * b1 - l * l - b2 * b2) / (-2 * l * b2);
  // cout << "\ttmp at triangle condition = " << tmp << endl;
  if (tmp > 1) {
    if ((tmp - 1) * (tmp - 1) < SING_TOL * SING_TOL)
      tmp = 1;
    else
      return SwivelSample{{1e10, 1e10}, {q7, q7}};
  } else if (tmp > 0.9 && n_fine > 0) {
    return theta_err_from_q7_explore_around_sing(
        q7, theta, i_E_O, k_E_O, i_6_O, n1_O, r_O7S_O, u_O7S_O, q7step, n_fine);
  }
  double actmp = acos(tmp);
  double alpha2 = beta2 + actmp;
  array<double, 3> k_C_O = {-r6[0] / l, -r6[1] / l, -r6[2] / l};
  array<double, 3> i_C_O = Cross(k_C_O, s6);
  tmp = Norm(i_C_O);
  i_C_O = {i_C_O[0] / tmp, i_C_O[1] / tmp, i_C_O[2] / tmp};
  array<double, 3> j_C_O;
  Cross_(k_C_O, i_C_O, j_C_O);
  double ry = s6[0] * j_C_O[0] + s6[1] * j_C_O[1] + s6[2] * j_C_O[2];
  double rz = s6[0] * k_C_O[0] + s6[1] * k_C_O[1] + s6[2] * k_C_O[2];
  double sa2, ca2;
  sa2 = sin(alpha2);
  ca2 = cos(alpha2);
  tmp = -rz * ca2 / (ry * sa2);
  // cout << "\ttmp at cone condition: " << tmp << endl;
  if (tmp * tmp > 1) {
    if (tmp * tmp < 1.2 && n_fine > 0) {
      // cout << "\texploring around current value of q7" << endl;
      //  errs = theta_err_from_q7_explore_around_sing(q7, theta, i_E_O, k_E_O, i_6_O, n1_O, r_O7S_O, u_O7S_O, q7step,
      //  n_fine);
      // cout << "\tbest q7 found: " << q7 * 180 / PI << " with errs: " << errs[0] << ", " << errs[1] << endl;
      //  return errs;
      return theta_err_from_q7_explore_around_sing(
          q7, theta, i_E_O, k_E_O, i_6_O, n1_O, r_O7S_O, u_O7S_O, q7step, n_fine);
    } else
      return SwivelSample{{1e15, 1e15}, {q7, q7}};
  }
  if (tmp * tmp > 0.8 && n_fine > 0) {
    // cout << "\texploring around current value of q7" << endl;
    //  errs = theta_err_from_q7_explore_around_sing(q7, theta, i_E_O, k_E_O, i_6_O, n1_O, r_O7S_O, u_O7S_O, q7step,
    //  n_fine);
    // cout << "\tbest q7 found: " << q7 * 180 / PI << " with errs: " << errs[0] << ", " << errs[1] <<endl;
    //  return errs;
    return theta_err_from_q7_explore_around_sing(
        q7, theta, i_E_O, k_E_O, i_6_O, n1_O, r_O7S_O, u_O7S_O, q7step, n_fine);
  }
  tmp = asin(tmp);
  double v[3] = {-sa2 * cos(tmp), -sa2 * sin(tmp), -ca2};
  array<array<double, 3>, 2> s5s;
  s5s[0] = {
      i_C_O[0] * v[0] + j_C_O[0] * v[1] + k_C_O[0] * v[2],
      i_C_O[1] * v[0] + j_C_O[1] * v[1] + k_C_O[1] * v[2],
      i_C_O[2] * v[0] + j_C_O[2] * v[1] + k_C_O[2] * v[2]};
  tmp = 2 * sa2 * cos(tmp);
  s5s[1] = {s5s[0][0] + tmp * i_C_O[0], s5s[0][1] + tmp * i_C_O[1], s5s[0][2] + tmp * i_C_O[2]};
  array<double, 3> s4, r4, n2_O;
  for (int i = 0; i < 2; i++) {
    s4 = Cross(s5s[i], r6);
    tmp = Norm(s4);
    s4 = {s4[0] / tmp, s4[1] / tmp, s4[2] / tmp};
    Cross_(s5s[i], s4, r4);
    r4 = {
        r6[0] - d5 * s5s[i][0] + a5 * r4[0], r6[1] - d5 * s5s[i][1] + a5 * r4[1], r6[2] - d5 * s5s[i][2] + a5 * r4[2]};
    Cross_(r_O7S_O, r4, n2_O);
    tmp = Dot(n2_O, s4);
    if (tmp < 0) n2_O = {-n2_O[0], -n2_O[1], -n2_O[2]};
    // franky vendoring patch: upstream squared this. The refinement in
    // franka_ik_swivel locates the minimum by intersecting the secant lines through
    // the two samples on each side, which is exact for the V-shaped |error| profile
    // and degenerates to "midpoint of the two central samples" for a squared one --
    // algebraically independent of the error values, so the refinement bought
    // nothing. Leaving it unsquared makes that interpolation do its job.
    // ERR_THRESH is expressed as an angle to match. See README.md.
    signed_errs[i] = wrapped_angle_difference(theta, signed_angle(n1_O, n2_O, u_O7S_O));
  }
  return SwivelSample{signed_errs, {q7, q7}};
}

void franka_ik_q7_one_sol(
    const double q7, const Eigen::Vector3d& i_E_O, const array<double, 3>& k_E_O, Eigen::Vector3d& i_6_O,
    const array<double, 3>& r_O7S_O, const unsigned int branch, array<array<double, 7>, 8>& qsols, unsigned int ind,
    const double q1_sing) {
  // returns the two solution related to one single branch of the IK with q7 as free variable. The results are stored in
  // qsols[s*ind] and qsols[2*ind+1]
  i_6_O = rotate_by_axis_angle(k_E_O, -(q7 - PI / 4), i_E_O);
  array<double, 3> s6 = Cross(k_E_O, i_6_O);
  array<double, 3> r6 = {r_O7S_O[0] - a7 * i_6_O[0], r_O7S_O[1] - a7 * i_6_O[1], r_O7S_O[2] - a7 * i_6_O[2]};
  double l = Norm(r6);
  double tmp = (b1 * b1 - l * l - b2 * b2) / (-2 * l * b2);
  // The exception tmp*tmp>1 was already handled during the sweep.
  // cout << "FINAL tmp triangle: " << tmp << endl;
  if (tmp > 1) tmp = 1;
  double actmp = acos(tmp);
  double alpha2 = beta2 + actmp;
  array<double, 3> k_C_O = {-r6[0] / l, -r6[1] / l, -r6[2] / l};
  array<double, 3> i_C_O = Cross(k_C_O, s6);
  tmp = Norm(i_C_O);
  i_C_O = {i_C_O[0] / tmp, i_C_O[1] / tmp, i_C_O[2] / tmp};
  array<double, 3> j_C_O = Cross(k_C_O, i_C_O);
  double ry = s6[0] * j_C_O[0] + s6[1] * j_C_O[1] + s6[2] * j_C_O[2];
  double rz = s6[0] * k_C_O[0] + s6[1] * k_C_O[1] + s6[2] * k_C_O[2];
  double sa2, ca2;
  sa2 = sin(alpha2);
  ca2 = cos(alpha2);
  tmp = -rz * ca2 / (ry * sa2);
  // The exception tmp*tmp>1 was already handled during the sweep.
  // cout << "FINAL tmp cone: " << tmp << endl;
  // cout << "cone variables: rz = " << rz << ", ry = " << ry << ", ca2 = " << ca2 << ", sa2 = " << sa2 << "" << endl;
  tmp = asin(tmp);
  double v[3] = {-sa2 * cos(tmp), -sa2 * sin(tmp), -ca2};
  array<double, 3> s5;
  s5 = {
      i_C_O[0] * v[0] + j_C_O[0] * v[1] + k_C_O[0] * v[2],
      i_C_O[1] * v[0] + j_C_O[1] * v[1] + k_C_O[1] * v[2],
      i_C_O[2] * v[0] + j_C_O[2] * v[1] + k_C_O[2] * v[2]};
  if (branch == 1) {
    tmp = 2 * sa2 * cos(tmp);
    s5 = {s5[0] + tmp * i_C_O[0], s5[1] + tmp * i_C_O[1], s5[2] + tmp * i_C_O[2]};
  }
  array<double, 3> s4, r4, s3, s2;
  array<double, 6> sol1;
  array<double, 3> sol2;
  s4 = Cross(s5, r6);
  tmp = Norm(s4);
  s4 = {s4[0] / tmp, s4[1] / tmp, s4[2] / tmp};
  r4 = Cross(s5, s4);
  r4 = {r6[0] - d5 * s5[0] + a5 * r4[0], r6[1] - d5 * s5[1] + a5 * r4[1], r6[2] - d5 * s5[2] + a5 * r4[2]};
  Eigen::Matrix3d R_tmp = R_axis_angle(s4, beta1);
  s3 = {
      R_tmp(0, 0) * r4[0] + R_tmp(0, 1) * r4[1] + R_tmp(0, 2) * r4[2],
      R_tmp(1, 0) * r4[0] + R_tmp(1, 1) * r4[1] + R_tmp(1, 2) * r4[2],
      R_tmp(2, 0) * r4[0] + R_tmp(2, 1) * r4[1] + R_tmp(2, 2) * r4[2]};
  tmp = Norm(s3);
  s3 = {s3[0] / tmp, s3[1] / tmp, s3[2] / tmp};
  tmp = s3[1] * s3[1] + s3[0] * s3[0];
  if (tmp > SHOULDER_SING_TOL * SHOULDER_SING_TOL)
    s2 = {-s3[1] / sqrt(tmp), s3[0] / sqrt(tmp), 0};
  else
    s2 = {sin(q1_sing), cos(q1_sing), 0};
  Eigen::Matrix<double, 3, 7> tmp_J = J_dir(s2, s3, s4, s5, s6, k_E_O);
  sol1 = q_from_J(tmp_J);
  tmp_J.col(1) = -1 * tmp_J.col(1);
  sol2 = q_from_low_J(tmp_J);
  qsols[2 * ind] = {sol1[0], sol1[1], sol1[2], sol1[3], sol1[4], sol1[5], q7};
  check_limits(qsols[2 * ind], 7);
  qsols[2 * ind + 1] = {
      sol2[0], sol2[1], sol2[2], qsols[2 * ind][3], qsols[2 * ind][4], qsols[2 * ind][5], qsols[2 * ind][6]};
  check_limits(qsols[2 * ind + 1], 3);
}

static double interpolate_swivel_q7(
    const unsigned int minimum, const unsigned int branch, const unsigned int n_points,
    const array<array<double, 2>, MAX_N_POINTS>& signed_errors, const array<array<double, 2>, MAX_N_POINTS>& q7s) {
  double q7_opt = q7s[minimum][branch];

  // A sign change brackets the requested swivel angle directly. Unlike the
  // old four-point V interpolation, this also works next to a sweep boundary
  // and when the samples two cells away did not pass the coarse error gate.
  if (n_points > 1) {
    unsigned int left;
    unsigned int right;
    if (minimum == 0) {
      left = 0;
      right = 1;
    } else if (minimum + 1 == n_points) {
      left = minimum - 1;
      right = minimum;
    } else if (fabs(signed_errors[minimum + 1][branch]) < fabs(signed_errors[minimum - 1][branch])) {
      left = minimum;
      right = minimum + 1;
    } else {
      left = minimum - 1;
      right = minimum;
    }

    const double x1 = q7s[left][branch];
    const double x2 = q7s[right][branch];
    const double signed1 = signed_errors[left][branch];
    const double signed2 = signed_errors[right][branch];
    if (fabs(signed1) < ERR_THRESH && fabs(signed2) < ERR_THRESH && x1 < x2 &&
        ((signed1 <= 0.0 && signed2 >= 0.0) || (signed1 >= 0.0 && signed2 <= 0.0)) && signed1 != signed2) {
      const double candidate = x1 - signed1 * (x2 - x1) / (signed2 - signed1);
      if (candidate >= x1 && candidate <= x2) return candidate;
    }
  }

  // If the signed coordinate is discontinuous here, retain GeoFIK's original
  // minimum-of-|error| refinement, generalized to the actual (possibly moved)
  // near-singularity sample positions rather than an assumed uniform grid.
  if (minimum > 1 && minimum < n_points - 2 && fabs(signed_errors[minimum - 2][branch]) < ERR_THRESH &&
      fabs(signed_errors[minimum + 2][branch]) < ERR_THRESH) {
    const unsigned int first =
        fabs(signed_errors[minimum + 1][branch]) < fabs(signed_errors[minimum - 1][branch]) ? minimum - 1 : minimum - 2;
    const double e0 = fabs(signed_errors[first][branch]);
    const double e1 = fabs(signed_errors[first + 1][branch]);
    const double e2 = fabs(signed_errors[first + 2][branch]);
    const double e3 = fabs(signed_errors[first + 3][branch]);
    const double x0 = q7s[first][branch];
    const double x1 = q7s[first + 1][branch];
    const double x2 = q7s[first + 2][branch];
    const double x3 = q7s[first + 3][branch];
    if (x0 < x1 && x1 < x2 && x2 < x3) {
      const double left_slope = (e1 - e0) / (x1 - x0);
      const double right_slope = (e3 - e2) / (x3 - x2);
      const double slope_difference = left_slope - right_slope;
      if (slope_difference != 0.0) {
        const double candidate = (e2 - e1 + left_slope * x1 - right_slope * x2) / slope_difference;
        if (candidate > x1 && candidate < x2) q7_opt = candidate;
      }
    }
  }
  return q7_opt;
}

struct SwivelMinima {
  array<array<unsigned int, 2>, 4> values;
  unsigned int count = 0;
};

static SwivelMinima find_swivel_minima(
    const array<array<double, 2>, MAX_N_POINTS>& signed_errors, const unsigned int n_points) {
  SwivelMinima result;
  // Process branch 0 before branch 1 to preserve GeoFIK's public solution order.
  for (unsigned int branch = 0; branch < 2; ++branch) {
    bool in_run = false;
    unsigned int minimum = 0;
    for (unsigned int i = 0; i < n_points; ++i) {
      if (fabs(signed_errors[i][branch]) < ERR_THRESH) {
        if (!in_run) {
          minimum = i;
          in_run = true;
        } else if (fabs(signed_errors[i][branch]) < fabs(signed_errors[minimum][branch])) {
          minimum = i;
        }
      } else if (in_run) {
        if (result.count < result.values.size()) result.values[result.count++] = {minimum, branch};
        in_run = false;
      }
    }
    if (in_run && result.count < result.values.size()) result.values[result.count++] = {minimum, branch};
  }
  return result;
}

unsigned int franka_ik_swivel(
    const array<double, 3>& r, const array<double, 9>& ROE, const double theta, array<array<double, 7>, 8>& qsols,
    const double q1_sing, const unsigned int n_points, const unsigned int n_fine_search) {
  // IK with swivel angle as free variable (numerical)
  // INPUT: r = r_EO_O, position of frame E in frame O
  //        ROE, orientation of frame E in frame O (row-first format)
  //        theta, swivel angle (see paper for geometric defninition)
  //        qsols, array to store 8 solutions
  //        q1_sing, emergency value of q1 in case of singularity at shoulder joints (type-1 singularity).
  //        n_points, number of points to discretise the range of q7.
  // OUTPUT: number of solutions found.
  // NOTATION:
  // ri = r_iS_O,
  // si - s_i_O
  array<double, 3> k_E_O = {ROE[2], ROE[5], ROE[8]};
  array<double, 3> r_O7S_O = {r[0] - dE * k_E_O[0], r[1] - dE * k_E_O[1], r[2] - d1 - dE * k_E_O[2]};
  double tmp = sqrt(r_O7S_O[1] * r_O7S_O[1] + r_O7S_O[0] * r_O7S_O[0]);
  if (tmp < SING_TOL) {
    geofik_log() << "ERROR: n1_O is undefined";
    for (int i = 0; i < 8; i++) fill(qsols[i].begin(), qsols[i].end(), NAN);
    return 0;
  }
  array<double, 3> n1_O = {r_O7S_O[1] / tmp, -r_O7S_O[0] / tmp, 0};
  Eigen::Vector3d i_E_O(ROE[0], ROE[3], ROE[6]);
  Eigen::Vector3d i_6_O;
  tmp = Norm(r_O7S_O);
  array<double, 3> u_O7S_O = {r_O7S_O[0] / tmp, r_O7S_O[1] / tmp, r_O7S_O[2] / tmp};
  const double wrapped_theta = atan2(sin(theta), cos(theta));
  double q7_step = (q_up[6] - q_low[6]) / (n_points - 1);
  array<array<double, 2>, MAX_N_POINTS> SignedErrs;
  array<array<double, 2>, MAX_N_POINTS> q7s;

  // cout << "\nn1_O = " << n1_O[0] << ", " << n1_O[1] << ", " << n1_O[2] << endl;

  for (unsigned int i = 0; i < n_points; i++) {
    const double q7 = q_low[6] + i * q7_step;
    // cout << "\nq7 = " << q7s[i]*180/PI << endl;
    const SwivelSample sample =
        theta_err_from_q7(q7, wrapped_theta, i_E_O, k_E_O, i_6_O, n1_O, r_O7S_O, u_O7S_O, q7_step, n_fine_search);
    SignedErrs[i] = sample.signed_errors;
    q7s[i] = sample.q7s;
  }

  const SwivelMinima minima = find_swivel_minima(SignedErrs, n_points);
  if (minima.count == 0) {
    for (int i = 0; i < 8; i++) fill(qsols[i].begin(), qsols[i].end(), NAN);
    return 0;
  }

  array<unsigned int, 2> m;
  for (unsigned int i = 0; i < minima.count; i++) {
    m = minima.values[i];
    const double q7_opt = interpolate_swivel_q7(m[0], m[1], n_points, SignedErrs, q7s);
    franka_ik_q7_one_sol(q7_opt, i_E_O, k_E_O, i_6_O, r_O7S_O, m[1], qsols, i, q1_sing);
  }
  for (int i = 2 * minima.count; i < 8; ++i) {
    fill(qsols[i].begin(), qsols[i].end(), NAN);
  }
  return 2 * minima.count;
}

double franka_swivel(const array<double, 7>& q) {
  // swivel angle for a configuration q
  array<Eigen::Matrix4d, 9> Ts;
  franka_fk_all_frames(Ts, q);
  array<double, 3> r4 = {Ts[3](0, 3), Ts[3](1, 3), Ts[3](2, 3) - d1};
  array<double, 3> r7 = {Ts[6](0, 3), Ts[6](1, 3), Ts[6](2, 3) - d1};
  array<double, 3> s4 = {Ts[3](0, 2), Ts[3](1, 2), Ts[3](2, 2)};
  double tmp = sqrt(r7[1] * r7[1] + r7[0] * r7[0]);
  if (tmp < SING_TOL) {
    geofik_log() << "ERROR: n1_O is undefined";
    return NAN;
  }
  array<double, 3> n1_O = {r7[1] / tmp, -r7[0] / tmp, 0};
  array<double, 3> n2_O;
  Cross_(r7, r4, n2_O);
  tmp = Dot(n2_O, s4);
  if (tmp < 0) n2_O = {-n2_O[0], -n2_O[1], -n2_O[2]};
  tmp = Norm(r7);
  return signed_angle(n1_O, n2_O, array<double, 3>{r7[0] / tmp, r7[1] / tmp, r7[2] / tmp});
}

// FUNCTIONS FOR JACOBIAN MATRIX ==========================================================================

unsigned int franka_J_ik_q7(
    const array<double, 3>& r, const array<double, 9>& ROE, const double q7,
    array<array<array<double, 6>, 7>, 8>& Jsols, array<array<double, 7>, 8>& qsols, const bool joint_angles,
    const char Jacobian_ee, const double q1_sing) {
  // IK to calculate Jacobian and joint angles with q7 as free variable.
  // INPUT: r = r_EO_O, position of frame E in frame O
  //        ROE, orientation of frame E in frame O (row-first format)
  //        q7, value of joint angle of joint 7
  //        Jsols, array to store 8 Jacobian solutions
  //        qsols, array to store 8 joint-angle solutions
  //        joint_angles, if false only Jacobians are returned
  //        Jacobian_ee, end-effector frame of the Jacobian, not the IK. Only 'E', 'F', '8' and '6' are supported.
  //        q1_sing, emergency value of q1 in case of singularity at shoulder joints (type-1 singularity).
  // OUTPUT: number of solutions found.
  // NOTATION:
  // ri = r_iS_O,
  // si - s_i_O,
  Eigen::Vector3d i_E_O(ROE[0], ROE[3], ROE[6]);
  array<double, 3> k_E_O = {ROE[2], ROE[5], ROE[8]};
  Eigen::Vector3d i_6_O = rotate_by_axis_angle(k_E_O, -(q7 - PI / 4), i_E_O);
  array<double, 3> s6;
  Cross_(k_E_O, i_6_O, s6);
  array<double, 3> r6 = {
      r[0] - dE * k_E_O[0] - a7 * i_6_O[0],
      r[1] - dE * k_E_O[1] - a7 * i_6_O[1],
      r[2] - d1 - dE * k_E_O[2] - a7 * i_6_O[2]};
  double l = Norm(r6);
  double tmp = (b1 * b1 - l * l - b2 * b2) / (-2 * l * b2);
  if (tmp > 1) {
    if ((tmp - 1) * (tmp - 1) < SING_TOL * SING_TOL) {
      tmp = 1;
    } else {
      geofik_log() << "ERROR: unable to assembly kinematic chain";
      for (int i = 0; i < 8; ++i) {
        fill(qsols[i].begin(), qsols[i].end(), NAN);
        for (auto& row : Jsols[i]) fill(row.begin(), row.end(), NAN);
      }
      return 0;
    }
  }
  double actmp = acos(tmp);
  double alpha2 = beta2 + actmp;
  array<double, 3> k_C_O = {-r6[0] / l, -r6[1] / l, -r6[2] / l};
  array<double, 3> i_C_O = Cross(k_C_O, s6);
  tmp = Norm(i_C_O);
  i_C_O = {i_C_O[0] / tmp, i_C_O[1] / tmp, i_C_O[2] / tmp};
  array<double, 3> j_C_O = Cross(k_C_O, i_C_O);
  double ry = s6[0] * j_C_O[0] + s6[1] * j_C_O[1] + s6[2] * j_C_O[2];
  double rz = s6[0] * k_C_O[0] + s6[1] * k_C_O[1] + s6[2] * k_C_O[2];
  array<array<double, 3>, 4> s5s;
  double sa2, ca2;
  int n_alphs = 1;
  unsigned int n_sols = 0;
  if (d3 + d5 < l && l < b1 + b2) n_alphs = 2;
  double v[3];
  for (int i = 0; i < n_alphs; i++) {
    // franky vendoring patch: same `continue`-skips-the-advance shape as in
    // franka_ik_q7. See README.md.
    alpha2 = (i == 0) ? (beta2 + actmp) : (beta2 - actmp);
    sa2 = sin(alpha2);
    ca2 = cos(alpha2);
    tmp = -rz * ca2 / (ry * sa2);
    if (tmp * tmp > 1) continue;
    tmp = asin(tmp);
    v[0] = -sa2 * cos(tmp);
    v[1] = -sa2 * sin(tmp);
    v[2] = -ca2;
    s5s[n_sols] = {
        i_C_O[0] * v[0] + j_C_O[0] * v[1] + k_C_O[0] * v[2],
        i_C_O[1] * v[0] + j_C_O[1] * v[1] + k_C_O[1] * v[2],
        i_C_O[2] * v[0] + j_C_O[2] * v[1] + k_C_O[2] * v[2]};
    tmp = 2 * sa2 * cos(tmp);
    // s5[n_sols+1] = s5s[n_sols] + (2*sa2*cos(tmp)*i_C_O);
    s5s[n_sols + 1] = {
        s5s[n_sols][0] + tmp * i_C_O[0], s5s[n_sols][1] + tmp * i_C_O[1], s5s[n_sols][2] + tmp * i_C_O[2]};
    n_sols += 2;
  }
  // Jsols.resize(2 * n_sols);
  // vector<array<double, 7>> sols;
  array<double, 6> sol1;
  array<double, 3> sol2;
  array<double, 3> s4, r4, s3, s2, s5;
  for (unsigned int i = 0; i < n_sols; i++) {
    s5 = s5s[i];
    Cross_(s5, r6, s4);
    tmp = Norm(s4);
    s4 = {s4[0] / tmp, s4[1] / tmp, s4[2] / tmp};
    // r4 = r6 - d5 * s5 + a5 * Cross(s5, s4);
    Cross_(s5, s4, r4);
    r4 = {r6[0] - d5 * s5[0] + a5 * r4[0], r6[1] - d5 * s5[1] + a5 * r4[1], r6[2] - d5 * s5[2] + a5 * r4[2]};
    // s3 = R_axis_angle(s4, beta1) * r4;
    Eigen::Matrix3d R_tmp = R_axis_angle(s4, beta1);
    s3 = {
        R_tmp(0, 0) * r4[0] + R_tmp(0, 1) * r4[1] + R_tmp(0, 2) * r4[2],
        R_tmp(1, 0) * r4[0] + R_tmp(1, 1) * r4[1] + R_tmp(1, 2) * r4[2],
        R_tmp(2, 0) * r4[0] + R_tmp(2, 1) * r4[1] + R_tmp(2, 2) * r4[2]};
    tmp = Norm(s3);
    s3 = {s3[0] / tmp, s3[1] / tmp, s3[2] / tmp};
    tmp = s3[1] * s3[1] + s3[0] * s3[0];
    if (tmp > SHOULDER_SING_TOL * SHOULDER_SING_TOL) {
      s2 = {-s3[1] / sqrt(tmp), s3[0] / sqrt(tmp), 0};
    } else {
      s2 = {sin(q1_sing), cos(q1_sing), 0};
    }
    save_J_sol(s2, s3, s4, s5, s6, k_E_O, r4, r6, r, Jsols, i, Jacobian_ee);
    if (joint_angles) {
      Eigen::Matrix<double, 3, 7> tmp_J = J_dir(s2, s3, s4, s5, s6, k_E_O);
      sol1 = q_from_J(tmp_J);
      tmp_J.col(1) = -1 * tmp_J.col(1);
      sol2 = q_from_low_J(tmp_J);
      qsols[2 * i] = {sol1[0], sol1[1], sol1[2], sol1[3], sol1[4], sol1[5], q7};
      check_limits(qsols[2 * i], 7);
      qsols[2 * i + 1] = {
          sol2[0], sol2[1], sol2[2], qsols[2 * i][3], qsols[2 * i][4], qsols[2 * i][5], qsols[2 * i][6]};
      check_limits(qsols[2 * i + 1], 3);
    }
  }
  for (int i = 2 * n_sols; i < 8; ++i) {
    for (auto& row : Jsols[i]) fill(row.begin(), row.end(), NAN);
  }
  for (int i = joint_angles ? 2 * n_sols : 0; i < 8; i++) fill(qsols[i].begin(), qsols[i].end(), NAN);
  return 2 * n_sols;
}

unsigned int franka_J_ik_q4(
    const array<double, 3>& r, const array<double, 9>& ROE, const double q4,
    array<array<array<double, 6>, 7>, 8>& Jsols, array<array<double, 7>, 8>& qsols, const bool joint_angles,
    const char Jacobian_ee, const double q1_sing, const double q7_sing) {
  // IK to calculate Jacobian and joint angles with q4 as free variable.
  // INPUT: r = r_EO_O, position of frame E in frame O
  //        ROE, orientation of frame E in frame O (row-first format)
  //        q4, value of joint angle of joint 4
  //        Jsols, array to store 8 Jacobian solutions
  //        qsols, array to store 8 joint-angle solutions
  //        joint_angles, if false only Jacobians are returned
  //        Jacobian_ee, end-effector frame of the Jacobian, not the IK. Only 'E', 'F', '8' and '6' are supported.
  //        q1_sing, emergency value of q1 in case of singularity at shoulder joints (type-1 singularity).
  //        q7_sing, emergency value of q7 in case S7 intersects S (type-2 singularity)
  // OUTPUT: number of solutions found.
  // NOTATION:
  // ri = r_iS_O,
  // si - s_i_O,
  array<double, 3> r_ES_O = {r[0], r[1], r[2] - d1};
  array<double, 3> tmp_v = {
      r_ES_O[1] * ROE[8] - r_ES_O[2] * ROE[5],
      r_ES_O[2] * ROE[2] - r_ES_O[0] * ROE[8],
      r_ES_O[0] * ROE[5] - r_ES_O[1] * ROE[2]};
  if (tmp_v[0] * tmp_v[0] + tmp_v[1] * tmp_v[1] + tmp_v[2] * tmp_v[2] < SING_TOL * SING_TOL)
    return franka_J_ik_q7(r, ROE, q7_sing, Jsols, qsols, joint_angles, Jacobian_ee, q1_sing);
  array<double, 3> r_O7S_O = {r_ES_O[0] - dE * ROE[2], r_ES_O[1] - dE * ROE[5], r_ES_O[2] - dE * ROE[8]};
  array<double, 3> r_O7S_E = {
      ROE[0] * r_O7S_O[0] + ROE[3] * r_O7S_O[1] + ROE[6] * r_O7S_O[2],
      ROE[1] * r_O7S_O[0] + ROE[4] * r_O7S_O[1] + ROE[7] * r_O7S_O[2],
      ROE[2] * r_O7S_O[0] + ROE[5] * r_O7S_O[1] + ROE[8] * r_O7S_O[2]};
  double alpha = q4 + beta1 + beta2 - PI;
  double lo2 = b1 * b1 + b2 * b2 - 2 * b1 * b2 * cos(alpha);
  double lp2 = lo2 - r_O7S_E[2] * r_O7S_E[2];
  if (lp2 < 0) {
    if (lp2 * lp2 < SING_TOL * SING_TOL)
      lp2 = 0;
    else {
      geofik_log() << "\nERROR: unable to assembly kinematic chain\n";
      for (int i = 0; i < 8; ++i) fill(qsols[i].begin(), qsols[i].end(), NAN);
      return 0;
    }
  }
  double gamma2 = beta2 + asin(b1 * sin(alpha) / sqrt(lo2));
  double cg2 = cos(gamma2), sg2 = sin(gamma2);
  double Lp2 = r_O7S_E[0] * r_O7S_E[0] + r_O7S_E[1] * r_O7S_E[1], phi = atan2(-r_O7S_E[1], -r_O7S_E[0]);
  double tmp = (Lp2 + a7 * a7 - lp2) / (2 * sqrt(Lp2) * a7);
  if (tmp > 1.0) {
    if ((tmp - 1) * (tmp - 1) < SING_TOL * SING_TOL)
      tmp = 1.0;
    else {
      geofik_log() << "\nERROR: unable to assembly kinematic chain\n";
      for (int i = 0; i < 8; ++i) {
        fill(qsols[i].begin(), qsols[i].end(), NAN);
      }
      return 0;
    }
  }
  double psi = acos(tmp), ry, rz;
  double q7s[2] = {-phi - psi - 3 * PI / 4, -phi + psi - 3 * PI / 4};
  double gammas[2] = {0, 0};
  size_t ind = 0;
  array<double, 3> s2, s3, s4, s5, s6, r4, r6, i_C_O, j_C_O, k_C_O;
  array<double, 3> s7 = {ROE[2], ROE[5], ROE[8]};
  array<double, 6> sol1;
  array<double, 3> sol2;
  for (auto q7 : q7s) {
    tmp_v = {cos(-q7 + 3 * PI / 4), sin(-q7 + 3 * PI / 4), 0};
    s6 = {
        ROE[0] * tmp_v[0] + ROE[1] * tmp_v[1],
        ROE[3] * tmp_v[0] + ROE[4] * tmp_v[1],
        ROE[6] * tmp_v[0] + ROE[7] * tmp_v[1]};
    tmp_v = {-a7 * cos(-q7 + PI / 4), -a7 * sin(-q7 + PI / 4), 0};
    r6 = {
        ROE[0] * tmp_v[0] + ROE[1] * tmp_v[1],
        ROE[3] * tmp_v[0] + ROE[4] * tmp_v[1],
        ROE[6] * tmp_v[0] + ROE[7] * tmp_v[1]};
    r6 = {r6[0] + r_O7S_O[0], r6[1] + r_O7S_O[1], r6[2] + r_O7S_O[2]};
    tmp = Norm(r6);
    k_C_O = {-r6[0] / tmp, -r6[1] / tmp, -r6[2] / tmp};
    Cross_(k_C_O, s6, i_C_O);
    tmp = Norm(i_C_O);
    i_C_O = {i_C_O[0] / tmp, i_C_O[1] / tmp, i_C_O[2] / tmp};
    Cross_(k_C_O, i_C_O, j_C_O);
    ry = s6[0] * j_C_O[0] + s6[1] * j_C_O[1] + s6[2] * j_C_O[2];
    rz = s6[0] * k_C_O[0] + s6[1] * k_C_O[1] + s6[2] * k_C_O[2];
    tmp = -rz * cg2 / (ry * sg2);
    if (tmp * tmp > 1) continue;
    tmp = asin(tmp);
    gammas[0] = tmp;
    gammas[1] = PI - tmp;
    for (auto gamma : gammas) {
      tmp_v = {-sg2 * cos(gamma), -sg2 * sin(gamma), -cg2};
      s5 = {
          i_C_O[0] * tmp_v[0] + j_C_O[0] * tmp_v[1] + k_C_O[0] * tmp_v[2],
          i_C_O[1] * tmp_v[0] + j_C_O[1] * tmp_v[1] + k_C_O[1] * tmp_v[2],
          i_C_O[2] * tmp_v[0] + j_C_O[2] * tmp_v[1] + k_C_O[2] * tmp_v[2]};
      Cross_(s5, r6, s4);
      tmp = Norm(s4);
      s4 = {s4[0] / tmp, s4[1] / tmp, s4[2] / tmp};
      Cross_(s5, s4, r4);
      r4 = {r6[0] - d5 * s5[0] + a5 * r4[0], r6[1] - d5 * s5[1] + a5 * r4[1], r6[2] - d5 * s5[2] + a5 * r4[2]};
      Eigen::Matrix3d R_tmp = R_axis_angle(s4, beta1);
      s3 = {
          R_tmp(0, 0) * r4[0] + R_tmp(0, 1) * r4[1] + R_tmp(0, 2) * r4[2],
          R_tmp(1, 0) * r4[0] + R_tmp(1, 1) * r4[1] + R_tmp(1, 2) * r4[2],
          R_tmp(2, 0) * r4[0] + R_tmp(2, 1) * r4[1] + R_tmp(2, 2) * r4[2]};
      tmp = Norm(s3);
      s3 = {s3[0] / tmp, s3[1] / tmp, s3[2] / tmp};
      tmp = s3[1] * s3[1] + s3[0] * s3[0];
      if (tmp > SHOULDER_SING_TOL * SHOULDER_SING_TOL)
        s2 = {-s3[1] / sqrt(tmp), s3[0] / sqrt(tmp), 0};
      else
        s2 = {sin(q1_sing), cos(q1_sing), 0};
      save_J_sol(s2, s3, s4, s5, s6, s7, r4, r6, r, Jsols, ind, Jacobian_ee);
      if (joint_angles) {
        Eigen::Matrix<double, 3, 7> tmp_J = J_dir(s2, s3, s4, s5, s6, s7);
        sol1 = q_from_J(tmp_J);
        tmp_J.col(1) = -1 * tmp_J.col(1);
        sol2 = q_from_low_J(tmp_J);
        qsols[2 * ind] = {sol1[0], sol1[1], sol1[2], sol1[3], sol1[4], sol1[5], q7};
        check_limits(qsols[2 * ind], 7);
        qsols[2 * ind + 1] = {
            sol2[0], sol2[1], sol2[2], qsols[2 * ind][3], qsols[2 * ind][4], qsols[2 * ind][5], qsols[2 * ind][6]};
        check_limits(qsols[2 * ind + 1], 3);
      }
      ind++;
    }
  }
  for (int i = 2 * ind; i < 8; ++i) {
    for (auto& row : Jsols[i]) fill(row.begin(), row.end(), NAN);
  }
  for (int i = joint_angles ? 2 * ind : 0; i < 8; i++) fill(qsols[i].begin(), qsols[i].end(), NAN);
  return 2 * ind;
}

unsigned int franka_J_ik_q6_parallel(
    const array<double, 3>& r, const array<double, 3>& r_ES_O, const array<double, 9>& ROE, const int sgn,
    array<array<array<double, 6>, 7>, 8>& Jsols, array<array<double, 7>, 8>& qsols, const bool joint_angles,
    const char Jacobian_ee, const double q1_sing) {
  // Parallel case of the Jacobian IK with q6 as free variable. Only called by franka_J_ik_q6(), not by the user.
  // INPUT: r, r_ES_O, ROE, sgn  = sign(cos(q6)), Jsols, qsols, joint_angles, Jacobian_ee, q1_sing.
  // OUTPUT: number of solutions found.
  // NOTATION:
  // ri = r_iS_O, i = 1,2,3,4,5,6,7
  // si = s_i_O
  // Q is a frame that is parallel to frame E and has origin at Q (Q is called E' in the paper)
  array<double, 3> s7 = {ROE[2], ROE[5], ROE[8]};
  array<double, 3> r_QS_O = {
      r_ES_O[0] + (-dE + sgn * d5) * s7[0], r_ES_O[1] + (-dE + sgn * d5) * s7[1], r_ES_O[2] + (-dE + sgn * d5) * s7[2]};
  array<double, 3> r_SQ_Q = {
      -ROE[0] * r_QS_O[0] - ROE[3] * r_QS_O[1] - ROE[6] * r_QS_O[2],
      -ROE[1] * r_QS_O[0] - ROE[4] * r_QS_O[1] - ROE[7] * r_QS_O[2],
      -ROE[2] * r_QS_O[0] - ROE[5] * r_QS_O[1] - ROE[8] * r_QS_O[2]};
  double tmp = b1 * b1 - r_SQ_Q[2] * r_SQ_Q[2];
  if (tmp * tmp < SING_TOL * SING_TOL) tmp = 0;
  if (tmp < 0) {
    geofik_log() << "ERROR: unable to assembly kinematic chain";
    for (int i = 0; i < 8; ++i) {
      fill(qsols[i].begin(), qsols[i].end(), NAN);
      for (auto& row : Jsols[i]) fill(row.begin(), row.end(), NAN);
    }
    return 0;
  }
  double lp = sqrt(tmp);
  array<double, 3> r_SpQ_Q = {r_SQ_Q[0], r_SQ_Q[1], 0};
  double l_SpQ = sqrt(r_SQ_Q[0] * r_SQ_Q[0] + r_SQ_Q[1] * r_SQ_Q[1]);
  double alphas[2], Ls[2];
  double q7;
  Ls[0] = a5 + lp, Ls[1] = a5 - lp;
  array<double, 3> tmp_v, r_O6pQ_Q, i_4_Q, r_O4Q_Q, s6_Q, r_O6Q_Q, s4_Q, s3_Q, s2, s3, s4, s5, s6, r4, r6;
  Eigen::Matrix<double, 3, 4> partial_J_Q, partial_J_O;
  Eigen::Matrix<double, 3, 2> rs;
  Eigen::Matrix3d ROQ;
  ROQ << ROE[0], ROE[1], ROE[2], ROE[3], ROE[4], ROE[5], ROE[6], ROE[7], ROE[8];
  const array<double, 3> k{{0, 0, 1}};
  array<double, 3> s5_Q{{0, 0, -1.0 * sgn}};
  int tmp_sgn;
  unsigned int ind = 0;
  array<double, 6> sol1;
  array<double, 3> sol2;
  for (auto L : Ls) {
    tmp = (-L * L + a7 * a7 + l_SpQ * l_SpQ) / (2 * a7 * l_SpQ);
    if ((tmp - 1) * (tmp - 1) < SING_TOL * SING_TOL)
      tmp = 1;
    else if ((tmp + 1) * (tmp + 1) < SING_TOL * SING_TOL)
      tmp = -1;
    if (tmp * tmp > 1) continue;
    alphas[0] = acos(tmp);
    alphas[1] = -acos(tmp);
    for (auto alpha : alphas) {
      rotate_by_axis_angle(k, alpha, r_SpQ_Q, r_O6pQ_Q);
      r_O6pQ_Q = {a7 * r_O6pQ_Q[0] / l_SpQ, a7 * r_O6pQ_Q[1] / l_SpQ, a7 * r_O6pQ_Q[2] / l_SpQ};
      i_4_Q = {r_SpQ_Q[0] - r_O6pQ_Q[0], r_SpQ_Q[1] - r_O6pQ_Q[1], r_SpQ_Q[2] - r_O6pQ_Q[2]};
      tmp = Norm(i_4_Q);
      tmp_sgn = L < 0 ? -1 : 1;
      i_4_Q = {tmp_sgn * i_4_Q[0] / tmp, tmp_sgn * i_4_Q[1] / tmp, tmp_sgn * i_4_Q[2] / tmp};
      r_O4Q_Q = {r_O6pQ_Q[0] + a5 * i_4_Q[0], r_O6pQ_Q[1] + a5 * i_4_Q[1], r_O6pQ_Q[2] + a5 * i_4_Q[2]};
      Cross_(r_O6pQ_Q, k, s6_Q);
      r_O6Q_Q = {r_O6pQ_Q[0], r_O6pQ_Q[1], r_O6pQ_Q[2] - sgn * d5};
      rs << r_O4Q_Q[0], r_O6Q_Q[0], r_O4Q_Q[1], r_O6Q_Q[1], r_O4Q_Q[2], r_O6Q_Q[2];
      rs = ROQ * rs;  // r_O4Q_O, r_O6Q_O
      r4 = {rs(0, 0) + r_QS_O[0], rs(1, 0) + r_QS_O[1], rs(2, 0) + r_QS_O[2]};
      r6 = {rs(0, 1) + r_QS_O[0], rs(1, 1) + r_QS_O[1], rs(2, 1) + r_QS_O[2]};
      Cross_(i_4_Q, s5_Q, s4_Q);
      tmp_v = {r_O4Q_Q[0] - r_SQ_Q[0], r_O4Q_Q[1] - r_SQ_Q[1], r_O4Q_Q[2] - r_SQ_Q[2]};
      rotate_by_axis_angle(s4_Q, beta1, tmp_v, s3_Q);
      tmp = Norm(s3_Q);
      // s3_Q = {s3_Q[0]/tmp,s3_Q[1]/tmp,s3_Q[2]/tmp};
      partial_J_Q << s3_Q[0] / tmp, s4_Q[0], s5_Q[0], s6_Q[0], s3_Q[1] / tmp, s4_Q[1], s5_Q[1], s6_Q[1], s3_Q[2] / tmp,
          s4_Q[2], s5_Q[2], s6_Q[2];
      partial_J_O = ROQ * partial_J_Q;
      s3 = {partial_J_O(0, 0), partial_J_O(1, 0), partial_J_O(2, 0)};
      s4 = {partial_J_O(0, 1), partial_J_O(1, 1), partial_J_O(2, 1)};
      s5 = {partial_J_O(0, 2), partial_J_O(1, 2), partial_J_O(2, 2)};
      s6 = {partial_J_O(0, 3), partial_J_O(1, 3), partial_J_O(2, 3)};
      q7 = atan2(r_O6pQ_Q[1], -r_O6pQ_Q[0]) + PI / 4;
      tmp = s3[1] * s3[1] + s3[0] * s3[0];
      if (tmp > SHOULDER_SING_TOL * SHOULDER_SING_TOL)
        s2 = {-s3[1] / sqrt(tmp), s3[0] / sqrt(tmp), 0};
      else
        s2 = {sin(q1_sing), cos(q1_sing), 0};
      save_J_sol(s2, s3, s4, s5, s6, s7, r4, r6, r, Jsols, ind, Jacobian_ee);
      if (joint_angles) {
        Eigen::Matrix<double, 3, 7> tmp_J = J_dir(s2, s3, s4, s5, s6, s7);
        sol1 = q_from_J(tmp_J);
        tmp_J.col(1) = -1 * tmp_J.col(1);
        sol2 = q_from_low_J(tmp_J);
        qsols[2 * ind] = {sol1[0], sol1[1], sol1[2], sol1[3], sol1[4], sol1[5], q7};
        check_limits(qsols[2 * ind], 7);
        qsols[2 * ind + 1] = {
            sol2[0], sol2[1], sol2[2], qsols[2 * ind][3], qsols[2 * ind][4], qsols[2 * ind][5], qsols[2 * ind][6]};
        check_limits(qsols[2 * ind + 1], 3);
      }
      ind++;
    }
  }
  for (int i = 2 * ind; i < 8; ++i) {
    for (auto& row : Jsols[i]) fill(row.begin(), row.end(), NAN);
  }
  for (int i = joint_angles ? 2 * ind : 0; i < 8; i++) fill(qsols[i].begin(), qsols[i].end(), NAN);
  return 2 * ind;
}

unsigned int franka_J_ik_q6(
    const array<double, 3>& r, const array<double, 9>& ROE, const double q6,
    array<array<array<double, 6>, 7>, 8>& Jsols, array<array<double, 7>, 8>& qsols, const bool joint_angles,
    const char Jacobian_ee, const double q1_sing, const double q7_sing) {
  // IK to calculate Jacobian and joint angles with q6 as free variable.
  // INPUT: r = r_EO_O, position of frame E in frame O
  //        ROE, orientation of frame E in frame O (row-first format)
  //        q6, value of joint angle of joint 6
  //        Jsols, array to store 8 Jacobian solutions
  //        qsols, array to store 8 joint-angle solutions
  //        joint_angles, if false only Jacobians are returned
  //        Jacobian_ee, end-effector frame of the Jacobian, not the IK. Only 'E', 'F', '8' and '6' are supported.
  //        q1_sing, emergency value of q1 in case of singularity at shoulder joints (type-1 singularity).
  //        q7_sing, emergency value of q7 in case S7 intersects S (type-2 singularity)
  // OUTPUT: number of solutions found.
  // NOTATION:
  // ri = r_iS_O,
  // si - s_i_O,
  array<double, 3> r_ES_O = {r[0], r[1], r[2] - d1};
  array<double, 3> tmp_v = {
      r_ES_O[1] * ROE[8] - r_ES_O[2] * ROE[5],
      r_ES_O[2] * ROE[2] - r_ES_O[0] * ROE[8],
      r_ES_O[0] * ROE[5] - r_ES_O[1] * ROE[2]};
  if (tmp_v[0] * tmp_v[0] + tmp_v[1] * tmp_v[1] + tmp_v[2] * tmp_v[2] < SING_TOL * SING_TOL)
    return franka_J_ik_q7(r, ROE, q7_sing, Jsols, qsols, joint_angles, Jacobian_ee, q1_sing);
  if (sin(q6) * sin(q6) < WRIST_PARALLEL_TOL * WRIST_PARALLEL_TOL)
    // PARALLEL CASE:
    return franka_J_ik_q6_parallel(
        r, r_ES_O, ROE, cos(q6) >= 0 ? 1 : -1, Jsols, qsols, joint_angles, Jacobian_ee, q1_sing);
  // NON-PARALLEL CASE:
  array<double, 3> s7 = {ROE[2], ROE[5], ROE[8]};
  double gamma1 = PI - q6;
  double cg1 = cos(gamma1);
  double sg1 = sin(gamma1);
  array<double, 3> r_O7S_O = {r_ES_O[0] - dE * ROE[2], r_ES_O[1] - dE * ROE[5], r_ES_O[2] - dE * ROE[8]};
  array<double, 3> r_PS_O = {
      r_O7S_O[0] + (a7 / tan(gamma1)) * s7[0],
      r_O7S_O[1] + (a7 / tan(gamma1)) * s7[1],
      r_O7S_O[2] + (a7 / tan(gamma1)) * s7[2]};
  double lP = Norm(r_PS_O);
  double lC = a7 / sg1;
  double Cx = -(ROE[0] * r_PS_O[0] + ROE[3] * r_PS_O[1] + ROE[6] * r_PS_O[2]);
  double Cy = -(ROE[1] * r_PS_O[0] + ROE[4] * r_PS_O[1] + ROE[7] * r_PS_O[2]);
  double Cz = -(ROE[2] * r_PS_O[0] + ROE[5] * r_PS_O[1] + ROE[8] * r_PS_O[2]);
  double c = sqrt(a5 * a5 + (lC + d5) * (lC + d5));
  double tmp = (-b1 * b1 + lP * lP + c * c) / (2 * lP * c);
  if (tmp > 1.0) {
    if ((tmp - 1) * (tmp - 1) < SING_TOL * SING_TOL)
      tmp = 1.0;
    else {
      geofik_log() << "ERROR: unable to assembly kinematic chain";
      for (int i = 0; i < 8; ++i) {
        fill(qsols[i].begin(), qsols[i].end(), NAN);
        for (auto& row : Jsols[i]) fill(row.begin(), row.end(), NAN);
      }
      return 0;
    }
  }
  double tau = acos(tmp);
  unsigned int n_gamma_sols = 1;
  if ((d3 + d5 + lC < lP) && (lP < b1 + c)) n_gamma_sols = 2;
  double gamma2s[2];
  if (d5 < -lC)
    gamma2s[0] = tau + atan(a5 / (d5 + lC)) + PI;
  else
    gamma2s[0] = tau + atan(a5 / (d5 + lC));
  if (n_gamma_sols > 1) gamma2s[1] = gamma2s[0] - 2 * tau;
  array<array<double, 3>, 4> s5s;
  double q7s[4];
  double d, u1, u2;
  unsigned int n_sols = 0;
  for (unsigned int i = 0; i < n_gamma_sols; i++) {
    d = lP * cos(gamma2s[i]);
    tmp = (d + Cz * cg1) / (sqrt(Cx * Cx * sg1 * sg1 + Cy * Cy * sg1 * sg1));
    if (tmp > 1) {
      if ((tmp - 1) * (tmp - 1) < TRIG_DOMAIN_TOL * TRIG_DOMAIN_TOL)
        tmp = 1;
      else
        continue;
    } else if (tmp < -1) {
      if ((tmp + 1) * (tmp + 1) < TRIG_DOMAIN_TOL * TRIG_DOMAIN_TOL)
        tmp = -1;
      else
        continue;
    }
    u1 = asin(tmp);
    u2 = atan2(Cx * sg1, Cy * sg1);
    q7s[n_sols] = 5 * PI / 4 - u1 + u2;
    tmp_v = {-sg1 * cos(u1 - u2), -sg1 * sin(u1 - u2), cg1};
    column_1s_times_vec(ROE, tmp_v, s5s[n_sols]);
    n_sols++;
    q7s[n_sols] = PI / 4 + u1 + u2;
    tmp_v = {-sg1 * cos(PI - u1 - u2), -sg1 * sin(PI - u1 - u2), cg1};
    column_1s_times_vec(ROE, tmp_v, s5s[n_sols]);
    n_sols++;
  }
  array<double, 3> s2, s3, s4, s6, r4, r6;
  array<double, 6> sol1;
  array<double, 3> sol2;
  unsigned int assembled_sols = 0;
  for (unsigned int i = 0; i < n_sols; i++) {
    r6 = {r_PS_O[0] - lC * s5s[i][0], r_PS_O[1] - lC * s5s[i][1], r_PS_O[2] - lC * s5s[i][2]};
    tmp_v = {r_O7S_O[0] - r6[0], r_O7S_O[1] - r6[1], r_O7S_O[2] - r6[2]};
    Cross_(s7, tmp_v, s6);
    tmp = Norm(s6);
    s6 = {s6[0] / tmp, s6[1] / tmp, s6[2] / tmp};
    Cross_(s5s[i], r6, s4);
    tmp = Norm(s4);
    s4 = {s4[0] / tmp, s4[1] / tmp, s4[2] / tmp};
    Cross_(s5s[i], s4, tmp_v);
    r4 = {
        r6[0] - d5 * s5s[i][0] + a5 * tmp_v[0],
        r6[1] - d5 * s5s[i][1] + a5 * tmp_v[1],
        r6[2] - d5 * s5s[i][2] + a5 * tmp_v[2]};
    tmp = Norm(r4);
    // See the joint-angle solver above: reject algebraic roots that cannot
    // assemble the fixed-length upper-arm link.
    if (fabs(tmp - b1) > SING_TOL) continue;
    rotate_by_axis_angle(s4, beta1, r4, s3);
    s3 = {s3[0] / tmp, s3[1] / tmp, s3[2] / tmp};
    tmp = s3[1] * s3[1] + s3[0] * s3[0];
    if (tmp > SHOULDER_SING_TOL * SHOULDER_SING_TOL)
      s2 = {-s3[1] / sqrt(tmp), s3[0] / sqrt(tmp), 0};
    else
      s2 = {sin(q1_sing), cos(q1_sing), 0};
    save_J_sol(s2, s3, s4, s5s[i], s6, s7, r4, r6, r, Jsols, assembled_sols, Jacobian_ee);
    if (joint_angles) {
      Eigen::Matrix<double, 3, 7> tmp_J = J_dir(s2, s3, s4, s5s[i], s6, s7);
      sol1 = q_from_J(tmp_J);
      tmp_J.col(1) = -1 * tmp_J.col(1);
      sol2 = q_from_low_J(tmp_J);
      qsols[2 * assembled_sols] = {sol1[0], sol1[1], sol1[2], sol1[3], sol1[4], sol1[5], q7s[i]};
      check_limits(qsols[2 * assembled_sols], 7);
      qsols[2 * assembled_sols + 1] = {
          sol2[0],
          sol2[1],
          sol2[2],
          qsols[2 * assembled_sols][3],
          qsols[2 * assembled_sols][4],
          qsols[2 * assembled_sols][5],
          qsols[2 * assembled_sols][6]};
      check_limits(qsols[2 * assembled_sols + 1], 3);
    }
    assembled_sols++;
  }
  for (int i = 2 * assembled_sols; i < 8; ++i) {
    for (auto& row : Jsols[i]) fill(row.begin(), row.end(), NAN);
  }
  for (int i = joint_angles ? 2 * assembled_sols : 0; i < 8; i++) fill(qsols[i].begin(), qsols[i].end(), NAN);
  return 2 * assembled_sols;
}

// FUNCTIONS FOR SWIVEL ANGLE (JACOBIAN)

void franka_J_ik_q7_one_sol(
    const double q7, const Eigen::Vector3d& i_E_O, const array<double, 3>& k_E_O, Eigen::Vector3d& i_6_O,
    const array<double, 3>& r_O7S_O, const array<double, 3>& r, array<array<array<double, 6>, 7>, 8>& Jsols,
    array<array<double, 7>, 8>& qsols, unsigned int ind, const bool joint_angles, const char Jacobian_ee,
    const unsigned int branch, const double q1_sing) {
  // returns the two solution related to one single branch of the IK with q7 as free variable. The results are stored in
  // Jsols[2*ind] and Jsols[2*ind+1]
  i_6_O = rotate_by_axis_angle(k_E_O, -(q7 - PI / 4), i_E_O);
  array<double, 3> s6 = Cross(k_E_O, i_6_O);
  array<double, 3> r6 = {r_O7S_O[0] - a7 * i_6_O[0], r_O7S_O[1] - a7 * i_6_O[1], r_O7S_O[2] - a7 * i_6_O[2]};
  double l = Norm(r6);
  double tmp = (b1 * b1 - l * l - b2 * b2) / (-2 * l * b2);
  // The exception tmp*tmp>1 was already handled during the sweep.
  if (tmp > 1) tmp = 1;
  double actmp = acos(tmp);
  double alpha2 = beta2 + actmp;
  array<double, 3> k_C_O = {-r6[0] / l, -r6[1] / l, -r6[2] / l};
  array<double, 3> i_C_O = Cross(k_C_O, s6);
  tmp = Norm(i_C_O);
  i_C_O = {i_C_O[0] / tmp, i_C_O[1] / tmp, i_C_O[2] / tmp};
  array<double, 3> j_C_O = Cross(k_C_O, i_C_O);
  double ry = s6[0] * j_C_O[0] + s6[1] * j_C_O[1] + s6[2] * j_C_O[2];
  double rz = s6[0] * k_C_O[0] + s6[1] * k_C_O[1] + s6[2] * k_C_O[2];
  double sa2, ca2;
  sa2 = sin(alpha2);
  ca2 = cos(alpha2);
  tmp = -rz * ca2 / (ry * sa2);
  // The exception tmp*tmp>1 was already handled during the sweep.
  tmp = asin(tmp);
  double v[3] = {-sa2 * cos(tmp), -sa2 * sin(tmp), -ca2};
  array<double, 3> s5;
  s5 = {
      i_C_O[0] * v[0] + j_C_O[0] * v[1] + k_C_O[0] * v[2],
      i_C_O[1] * v[0] + j_C_O[1] * v[1] + k_C_O[1] * v[2],
      i_C_O[2] * v[0] + j_C_O[2] * v[1] + k_C_O[2] * v[2]};
  if (branch == 1) {
    tmp = 2 * sa2 * cos(tmp);
    s5 = {s5[0] + tmp * i_C_O[0], s5[1] + tmp * i_C_O[1], s5[2] + tmp * i_C_O[2]};
  }
  array<double, 3> s4, r4, s3, s2;
  array<double, 6> sol1;
  array<double, 3> sol2;
  s4 = Cross(s5, r6);
  tmp = Norm(s4);
  s4 = {s4[0] / tmp, s4[1] / tmp, s4[2] / tmp};
  r4 = Cross(s5, s4);
  r4 = {r6[0] - d5 * s5[0] + a5 * r4[0], r6[1] - d5 * s5[1] + a5 * r4[1], r6[2] - d5 * s5[2] + a5 * r4[2]};
  Eigen::Matrix3d R_tmp = R_axis_angle(s4, beta1);
  s3 = {
      R_tmp(0, 0) * r4[0] + R_tmp(0, 1) * r4[1] + R_tmp(0, 2) * r4[2],
      R_tmp(1, 0) * r4[0] + R_tmp(1, 1) * r4[1] + R_tmp(1, 2) * r4[2],
      R_tmp(2, 0) * r4[0] + R_tmp(2, 1) * r4[1] + R_tmp(2, 2) * r4[2]};
  tmp = Norm(s3);
  s3 = {s3[0] / tmp, s3[1] / tmp, s3[2] / tmp};
  tmp = s3[1] * s3[1] + s3[0] * s3[0];
  if (tmp > SHOULDER_SING_TOL * SHOULDER_SING_TOL)
    s2 = {-s3[1] / sqrt(tmp), s3[0] / sqrt(tmp), 0};
  else
    s2 = {sin(q1_sing), cos(q1_sing), 0};
  save_J_sol(s2, s3, s4, s5, s6, k_E_O, r4, r6, r, Jsols, ind, Jacobian_ee);
  if (joint_angles) {
    Eigen::Matrix<double, 3, 7> tmp_J = J_dir(s2, s3, s4, s5, s6, k_E_O);
    sol1 = q_from_J(tmp_J);
    tmp_J.col(1) = -1 * tmp_J.col(1);
    sol2 = q_from_low_J(tmp_J);
    qsols[2 * ind] = {sol1[0], sol1[1], sol1[2], sol1[3], sol1[4], sol1[5], q7};
    check_limits(qsols[2 * ind], 7);
    qsols[2 * ind + 1] = {
        sol2[0], sol2[1], sol2[2], qsols[2 * ind][3], qsols[2 * ind][4], qsols[2 * ind][5], qsols[2 * ind][6]};
    check_limits(qsols[2 * ind + 1], 3);
  }
}

unsigned int franka_J_ik_swivel(
    const array<double, 3>& r, const array<double, 9>& ROE, const double theta,
    array<array<array<double, 6>, 7>, 8>& Jsols, array<array<double, 7>, 8>& qsols, const bool joint_angles,
    const char Jacobian_ee, const double q1_sing, const unsigned int n_points, const unsigned int n_fine_search) {
  // IK to calculate Jacobian and joint angles with swivel angle as free variable (numerical).
  // INPUT: r = r_EO_O, position of frame E in frame O
  //        ROE, orientation of frame E in frame O (row-first format)
  //        theta, swivel angle (see paper for gemetric definition)
  //        Jsols, array to store 8 Jacobian solutions
  //        qsols, array to store 8 joint-angle solutions
  //        joint_angles, if false only Jacobians are returned
  //        Jacobian_ee, end-effector frame of the Jacobian, not the IK. Only 'E', 'F', '8' and '6' are supported.
  //        q1_sing, emergency value of q1 in case of singularity at shoulder joints (type-1 singularity).
  //        n_points, number of points to descritise the range of q7
  // OUTPUT: number of solutions found.
  // NOTATION:
  // ri = r_iS_O,
  // si - s_i_O,
  array<double, 3> k_E_O = {ROE[2], ROE[5], ROE[8]};
  // r_O7S_O = r_EO_O + r_OS_O + r_O7E_O = r_EO_O - (0,0,d1) - dE*k_E_O
  array<double, 3> r_O7S_O = {r[0] - dE * k_E_O[0], r[1] - dE * k_E_O[1], r[2] - d1 - dE * k_E_O[2]};
  double tmp = sqrt(r_O7S_O[1] * r_O7S_O[1] + r_O7S_O[0] * r_O7S_O[0]);
  if (tmp < SING_TOL) {
    geofik_log() << "ERROR: n1_O is undefined";
    for (int i = 0; i < 8; ++i) {
      fill(qsols[i].begin(), qsols[i].end(), NAN);
      for (auto& row : Jsols[i]) fill(row.begin(), row.end(), NAN);
    }
    return 0;
  }
  array<double, 3> n1_O = {r_O7S_O[1] / tmp, -r_O7S_O[0] / tmp, 0};
  Eigen::Vector3d i_E_O(ROE[0], ROE[3], ROE[6]);
  Eigen::Vector3d i_6_O;
  tmp = Norm(r_O7S_O);
  array<double, 3> u_O7S_O = {r_O7S_O[0] / tmp, r_O7S_O[1] / tmp, r_O7S_O[2] / tmp};
  const double wrapped_theta = atan2(sin(theta), cos(theta));
  double q7_step = (q_up[6] - q_low[6]) / (n_points - 1);
  array<array<double, 2>, MAX_N_POINTS> SignedErrs;
  array<array<double, 2>, MAX_N_POINTS> q7s;
  for (unsigned int i = 0; i < n_points; i++) {
    const double q7 = q_low[6] + i * q7_step;
    const SwivelSample sample =
        theta_err_from_q7(q7, wrapped_theta, i_E_O, k_E_O, i_6_O, n1_O, r_O7S_O, u_O7S_O, q7_step, n_fine_search);
    SignedErrs[i] = sample.signed_errors;
    q7s[i] = sample.q7s;
  }

  const SwivelMinima minima = find_swivel_minima(SignedErrs, n_points);
  if (minima.count == 0) {
    for (int i = 0; i < 8; ++i) {
      fill(qsols[i].begin(), qsols[i].end(), NAN);
      for (auto& row : Jsols[i]) fill(row.begin(), row.end(), NAN);
    }
    return 0;
  }

  array<unsigned int, 2> m;
  for (unsigned int i = 0; i < minima.count; i++) {
    m = minima.values[i];
    const double q7_opt = interpolate_swivel_q7(m[0], m[1], n_points, SignedErrs, q7s);
    franka_J_ik_q7_one_sol(
        q7_opt, i_E_O, k_E_O, i_6_O, r_O7S_O, r, Jsols, qsols, i, joint_angles, Jacobian_ee, m[1], q1_sing);
  }
  for (int i = 2 * minima.count; i < 8; ++i) {
    for (auto& row : Jsols[i]) fill(row.begin(), row.end(), NAN);
  }
  for (int i = joint_angles ? 2 * minima.count : 0; i < 8; i++) fill(qsols[i].begin(), qsols[i].end(), NAN);
  return 2 * minima.count;
}
