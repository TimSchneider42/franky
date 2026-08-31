// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Pablo Lopez-Custodio, 2026 Nick Walker

/**
 * @file    nanogeofik.cpp
 * @brief   functions for the IK of the Franka arm.
 *
 * @details notation:
 *          - `r_PQ_O`: position vector of point P with respect to point Q
 * measured in frame O.
 *          - `ROF`: Rotation matrix representing the orientation of a frame F
 * with respect to frame O.
 *
 * @author  Pablo Lopez-Custodio, Nick Walker
 * @date    2026-08-25
 * @version 2.0
 */

#include "nanogeofik.h"

#include <algorithm>
#include <cmath>
#include <iostream>

// The swivel sweep ships an AVX2/AVX-512 batch kernel
// alongside the portable one below, selected once per call by CPU feature
// detection.
#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
#include <immintrin.h>
#define NANOGEOFIK_HAVE_AVX2_SWIVEL 1
#endif

// The upstream header carried a global `using namespace
// std;`. It was removed from the header (to avoid leaking std into every
// includer) and moved here, so the implementation below compiles unchanged.
using namespace std;

#ifdef NANOGEOFIK_VERBOSE
static std::ostream& nanogeofik_log() { return std::cerr; }
#else
static std::ostream& nanogeofik_log() {
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
constexpr double inv_a7 = 1.0 / a7;
// dE =  0.107 + 0.1034
constexpr double dE = 0.2104;
// b1 = sqrt(d3*d3 + a4*a4)
constexpr double b1 = 0.3265918706887849;
constexpr double inv_b1 = 1.0 / b1;
// b2 = sqrt(d5*d5 + a5*a5)
constexpr double b2 = 0.39276233271534583;
// beta1 = arctan(a4/d3)
constexpr double beta1 = 0.25537561488738186;
// Exact trigonometric values implied by beta1 = atan(a4 / d3).
//
// Beta1 = atan(a4 / d3) is a fixed geometric constant.  Every geometric
// branch rotates r4 about s4 by this fixed angle, and passing the angle to the
// general rotate_by_axis_angle made every solve evaluate sincos at run time
// (the callee is not inlined, so the compiler could not fold it).
constexpr double sin_beta1 = a4 / b1;
constexpr double cos_beta1 = d3 / b1;
// beta2 = arctan(a5/d5)
constexpr double beta2 = 0.21162680876562978;
// Exact trigonometric values implied by beta2 = atan(a5 / d5).
constexpr double sin_beta2 = a5 / b2;
constexpr double cos_beta2 = d5 / b2;

// Tolerance for entering singularity mode.
constexpr double SING_TOL = 1e-5;

// The shoulder fallback divides by the horizontal projection of the third joint
// axis. Double precision remains well-conditioned far closer to zero than the
// geometric assembly tests above; using their 1e-5 tolerance here needlessly
// replaced valid near-singular solutions with an arbitrary q1 value.
constexpr double SHOULDER_SING_TOL = 1e-8;

// The closed-form q6 construction divides by sin(q6): its cone step
// subtracts two O(a7/sin(q6)) quantities whose difference is O(sin(q6)), so
// below |sin(q6)| ~ SING_TOL it carries only ~5 significant digits while the
// parallel construction stays exact. The parallel branch therefore runs
// first across the whole sliver; it can fail to assemble for individual
// poses, which fall through to the closed form so coverage never regresses.
constexpr double WRIST_PARALLEL_TOL = 1e-5;

// Inverse-trig domain clamps are for floating-point overshoot only. A looser
// geometric tolerance admits roots that do not assemble into the requested arm.
constexpr double TRIG_DOMAIN_TOL = 1e-8;

static bool clamp_trig_roundoff(double& value) {
  if (value > 1.0) {
    if (value - 1.0 > TRIG_DOMAIN_TOL) return false;
    value = 1.0;
  } else if (value < -1.0) {
    if (-1.0 - value > TRIG_DOMAIN_TOL) return false;
    value = -1.0;
  }
  return true;
}

// A reconstructed joint angle can land a few ulps outside an inclusive hardware
// limit even when the source configuration lies exactly on that limit. Keep the
// tolerance far below any mechanically meaningful angle and clamp only that
// round-off-sized overshoot back to the boundary.
constexpr double JOINT_LIMIT_TOL = 1e-10;
constexpr double TWO_PI = 2.0 * PI;

// error threshold for swivel angle solver
// Upstream squared the swivel error and used 0.002 here
// ("slightly smaller than (3deg)*(3deg)"). The error is no longer squared (see
// theta_err_from_q7), so this is the same acceptance band expressed as an
// angle: sqrt(0.002) = 0.0447 rad, slightly under 3 deg. See README.md.
constexpr double ERR_THRESH = 0.0447;  // slightly smaller than 3deg
const double TAN_ERR_THRESH = tan(ERR_THRESH);
// max number of points in discretisation for swivel angle solver
constexpr unsigned int MAX_N_POINTS = 1000;
// Fast, branch-free atan2.
//
// The joint-angle reconstruction evaluates six atan2 calls per assembled arm
// (and up to four arms per solve), and the swivel sweep evaluates one per
// accepted sample, which made libm's atan2 roughly 40% of every direct-solver
// call.  glibc's atan2 is correctly rounded to under one ulp, which it buys
// with several data-dependent branches and a double-double fallback path.  The
// solver does not need that: a joint angle carrying a couple of ulps of error
// perturbs the reconstructed pose by ~1e-16 m, four orders of magnitude below
// the tightest tolerance anything here is checked against.
//
// So use the classic reduction instead.  Fold the argument into the first
// octant, fold [tan(pi/8), 1] onto [-tan(pi/8), 0] via
// atan(t) = pi/4 + atan((num - den) / (num + den)) so a single division and a
// single polynomial cover the quadrant, then evaluate FDLIBM's degree-11
// minimax for atan(u)/u (valid to under one ulp for |u| <= 7/16, comfortably
// wider than the tan(pi/8) = 0.4143 the folding produces).  Measured against
// glibc over 4e6 random and 8e6 swept argument pairs the result is within 2
// ulp, and every IEEE-754 special case (signed zeros, infinities, NaNs,
// subnormals) agrees exactly.
//
// Written without branches so it stays cheap when called on unpredictable data
// and so a run of independent calls can vectorize.
namespace nanogeofik_atan {

// FDLIBM aT[]: atan(u) = u - u * (odd(u^2) + even(u^2)), |u| <= 7/16.
constexpr double kA0 = 3.33333333333329318027e-01;
constexpr double kA1 = -1.99999999998764832476e-01;
constexpr double kA2 = 1.42857142725034663711e-01;
constexpr double kA3 = -1.11111104054623557880e-01;
constexpr double kA4 = 9.09088713343650656196e-02;
constexpr double kA5 = -7.69187620504482999495e-02;
constexpr double kA6 = 6.66107313738753120669e-02;
constexpr double kA7 = -5.83357013379057348645e-02;
constexpr double kA8 = 4.97687799461593236017e-02;
constexpr double kA9 = -3.65315727442169155270e-02;
constexpr double kA10 = 1.62858201153657823623e-02;

constexpr double kTanPiOver8 = 0.41421356237309503;
// Each constant is split so the offset it contributes is exact to ~1e-33.
constexpr double kPiOver4Hi = 0.78539816339744828;
constexpr double kPiOver4Lo = 3.06161699786838302e-17;
constexpr double kPiOver2Hi = 1.57079632679489656;
constexpr double kPiOver2Lo = 6.12323399573676604e-17;
constexpr double kPiHi = 3.14159265358979312;
constexpr double kPiLo = 1.22464679914735321e-16;

inline double atan_octant(const double u) {
  const double z = u * u;
  const double w = z * z;
  const double odd =
      z * (kA0 + w * (kA2 + w * (kA4 + w * (kA6 + w * (kA8 + w * kA10)))));
  const double even = w * (kA1 + w * (kA3 + w * (kA5 + w * (kA7 + w * kA9))));
  return u - u * (odd + even);
}

// Which quadrant/octant corrections the reduced value still needs.
constexpr unsigned char kFolded = 1u;  // add pi/4 (the [tan(pi/8), 1] fold)
constexpr unsigned char kSteep = 2u;   // reflect through pi/4: |y| > |x|
constexpr unsigned char kNegativeX =
    4u;  // reflect through pi/2: sign bit of x set
constexpr unsigned char kNotANumber = 8u;  // an operand was NaN

struct AtanReduction {
  double u;  // |u| <= tan(pi/8)
  unsigned char octant;
};

// Everything about atan2 except the polynomial: fold the argument into the
// first octant and record what has to be undone afterwards.  Split out so that
// a run of calls can evaluate its polynomials in one vectorized loop -- the
// polynomial is most of the work and, unlike this reduction, contains no
// selects at all.
inline AtanReduction atan2_reduce(const double y, const double x) {
  const double ax = fabs(x);
  const double ay = fabs(y);
  const bool steep = ax < ay;
  const double num = steep ? ax : ay;  // min(|x|, |y|)
  const double den = steep ? ay : ax;  // max(|x|, |y|)
  // Fold the upper half of num/den in [0, 1] onto [-tan(pi/8), 0] using
  // atan(t) = pi/4 + atan((num - den) / (num + den)), so a single division and
  // a single polynomial cover the whole quadrant.
  bool folded = num > kTanPiOver8 * den;
  // Halving keeps num + den finite for huge operands.  Folding implies
  // num > 0.41 * den, so with den > 1 both are normal and halving is exact;
  // below that num + den cannot overflow, so leave the operands alone and keep
  // subnormal ratios exact.
  const bool halve = folded & (den > 1.0);
  const double a = halve ? 0.5 * num : num;
  const double b = halve ? 0.5 * den : den;
  double u = (folded ? a - b : a) / (folded ? a + b : b);
  // A NaN ratio means num == den == 0 (true ratio 0) or num == den == inf
  // (true ratio 1, which is the folded representation with u == 0).
  const bool degenerate = !(u == u);
  u = degenerate ? 0.0 : u;
  folded = degenerate ? den != 0.0 : folded;
  AtanReduction reduction;
  reduction.u = u;
  reduction.octant = static_cast<unsigned char>(
      (folded ? kFolded : 0u) | (steep ? kSteep : 0u) |
      // The sign bit of x, not x < 0: atan2(+-0, -0.0) is +-pi.  Taken via
      // copysign because std::signbit's integer result is awkward for the
      // compiler to keep alongside the doubles.
      (copysign(1.0, x) < 0.0 ? kNegativeX : 0u) |
      // The degenerate-ratio fixup above must not rescue a NaN operand.
      (((x != x) | (y != y)) ? kNotANumber : 0u));
  return reduction;
}

// Undo the octant reduction.  Each pi constant is split so the offset it
// contributes stays exact to about 1e-33.
inline double atan2_finish(double r, const unsigned char octant,
                           const double y) {
  if (octant & kFolded) r = (kPiOver4Hi + r) + kPiOver4Lo;
  if (octant & kSteep) r = (kPiOver2Hi - r) + kPiOver2Lo;
  if (octant & kNegativeX) r = (kPiHi - r) + kPiLo;
  r = copysign(r, y);
  return (octant & kNotANumber) ? NAN : r;
}

inline double atan2(const double y, const double x) {
  const AtanReduction reduction = atan2_reduce(y, x);
  return atan2_finish(atan_octant(reduction.u), reduction.octant, y);
}

}  // namespace nanogeofik_atan

// Branch-free sincos for the root scans.
//
// The q5 root scan evaluates its locking equation at several hundred
// abscissae per solve, and every one of them needs both cos(delta) and
// sin(delta).  Routed through libm that is the single largest line in the
// solver's profile: glibc's sincos is correctly rounded, which it pays for
// with data-dependent branches and a multi-precision reduction path, and
// those branches also stop the surrounding loop from vectorizing.
//
// The scan does not need correct rounding.  Its abscissae are angles in
// |delta| < 4 (the range of pi/4 - q7 over joint 7's limits), so a single
// Cody-Waite step against a two-part pi/2 reduces them exactly: pio2_hi
// carries 33 significant bits, so k*pio2_hi is exact for the |k| <= 3 this
// range produces, and the neglected third part contributes below 1e-20.
// FDLIBM's minimax kernels then cover |r| <= pi/4.  Measured against glibc
// over 4e7 abscissae spanning the scan range the worst absolute error is one
// machine epsilon in both outputs, four orders of magnitude below the
// tightest tolerance the solver checks a residual against, and the unit
// circle closes to 2 eps.  Written without branches so a run of independent
// calls vectorizes.
namespace nanogeofik_sincos {

constexpr double kPio2Hi = 1.57079632673412561417e+00;
constexpr double kPio2Lo = 6.07710050650619224932e-11;
constexpr double kTwoOverPi = 6.36619772367581382433e-01;
// FDLIBM __kernel_sin: sin(r) = r + r^3 * poly(r^2), |r| <= pi/4.
constexpr double kS1 = -1.66666666666666324348e-01;
constexpr double kS2 = 8.33333333332248946124e-03;
constexpr double kS3 = -1.98412698298579493134e-04;
constexpr double kS4 = 2.75573137070700676789e-06;
constexpr double kS5 = -2.50507602534068634195e-08;
constexpr double kS6 = 1.58969099521155010221e-10;
// FDLIBM __kernel_cos: cos(r) = 1 - r^2/2 + r^4 * poly(r^2), |r| <= pi/4.
constexpr double kC1 = 4.16666666666666019037e-02;
constexpr double kC2 = -1.38888888888741095749e-03;
constexpr double kC3 = 2.48015872894767294178e-05;
constexpr double kC4 = -2.75573143513906633035e-07;
constexpr double kC5 = 2.08757232129817482790e-09;
constexpr double kC6 = -1.13596475577881948265e-11;

// Valid for any |x| where k = rint(x*2/pi) stays small enough that k*pio2_hi
// is exact, which covers every angle the solvers feed it by a wide margin.
static inline void sincos(const double x, double& sin_out, double& cos_out) {
  const double kf = std::rint(x * kTwoOverPi);
  const double r = (x - kf * kPio2Hi) - kf * kPio2Lo;
  const double z = r * r;
  const double sp =
      kS1 + z * (kS2 + z * (kS3 + z * (kS4 + z * (kS5 + z * kS6))));
  const double sr = r + r * z * sp;
  const double cp =
      kC1 + z * (kC2 + z * (kC3 + z * (kC4 + z * (kC5 + z * kC6))));
  const double cr = (1.0 - 0.5 * z) + z * z * cp;
  // Quadrant k mod 4 reads (sin, cos) off (+sr,+cr), (+cr,-sr), (-sr,-cr),
  // (-cr,+sr): odd quadrants swap the kernels, and each output changes sign
  // on its own pair of quadrants.
  const long long k = static_cast<long long>(kf);
  const bool swap = (k & 1) != 0;
  const bool neg_sin = (k & 2) != 0;
  const bool neg_cos = ((k + 1) & 2) != 0;
  const double s0 = swap ? cr : sr;
  const double c0 = swap ? sr : cr;
  sin_out = neg_sin ? -s0 : s0;
  cos_out = neg_cos ? -c0 : c0;
}

}  // namespace nanogeofik_sincos

#if defined(__GNUC__) || defined(__clang__)
#define NANOGEOFIK_NOINLINE __attribute__((noinline))
#else
#define NANOGEOFIK_NOINLINE
#endif

// Runtime CPU dispatch for the public solvers.  The hot kernels are
// straight-line double arithmetic that the compiler vectorizes; under the
// baseline x86-64 ISA it only reaches SSE2's two doubles per vector, and no
// FMA.  Cloning each entry point for "avx2,fma" and letting every internal
// helper inline into the clones gives the same operations four-wide with
// fused multiply-adds on CPUs that support them, while the default clone
// keeps bit-identical-to-before behavior everywhere else.  Dispatch happens
// once per public call through the ELF IFUNC resolver, which is negligible
// next to even the cheapest solve.  Results may differ by at most an ulp
// between clones where a mul/add pair contracts into an fma.
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__) && \
    defined(__ELF__)
#define NANOGEOFIK_TARGET_CLONES \
  __attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))
#else
#define NANOGEOFIK_TARGET_CLONES
#endif

// Only a small fraction of sweep samples need an actual angle from the scaled
// error gate below; keep that cold atan2 tail out of the hot sweep loop.
static NANOGEOFIK_NOINLINE double nanogeofik_swivel_atan2(const double y,
                                                          const double x) {
  return nanogeofik_atan::atan2(y, x);
}

static inline const JointLimits& resolve_limits(const SolverTuning& tuning) {
  if (tuning.custom_limits != nullptr) {
    return *tuning.custom_limits;
  }
  switch (tuning.limit_preset) {
    case LimitPreset::Panda:
      return kPandaJointLimits;
    case LimitPreset::FR3:
      return kFr3JointLimits;
    case LimitPreset::Custom:
    case LimitPreset::None:
      return kPandaJointLimits;
  }
  return kPandaJointLimits;
}

static array<double, 3> Cross(const array<double, 3>& u,
                              const array<double, 3>& v) {
  return array<double, 3>{u[1] * v[2] - v[1] * u[2], v[0] * u[2] - u[0] * v[2],
                          u[0] * v[1] - v[0] * u[1]};
}

static array<double, 3> Cross(const array<double, 3>& u,
                              const Eigen::Vector3d& v) {
  return array<double, 3>{u[1] * v[2] - v[1] * u[2], v[0] * u[2] - u[0] * v[2],
                          u[0] * v[1] - v[0] * u[1]};
}

static void Cross_(const array<double, 3>& u, const Eigen::Vector3d& v,
                   array<double, 3>& w) {
  w[0] = u[1] * v[2] - v[1] * u[2];
  w[1] = v[0] * u[2] - u[0] * v[2];
  w[2] = u[0] * v[1] - v[0] * u[1];
}

static void Cross_(const array<double, 3>& u, const array<double, 3>& v,
                   array<double, 3>& w) {
  w[0] = u[1] * v[2] - v[1] * u[2];
  w[1] = v[0] * u[2] - u[0] * v[2];
  w[2] = u[0] * v[1] - v[0] * u[1];
}

static double Dot(const array<double, 3>& u, const array<double, 3>& v) {
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

static double Norm(const array<double, 3>& u) {
  return sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
}

static void save_J_sol(const array<double, 3>& s2, const array<double, 3>& s3,
                       const array<double, 3>& s4, const array<double, 3>& s5,
                       const array<double, 3>& s6, const array<double, 3>& s7,
                       const array<double, 3>& r4, const array<double, 3>& r5,
                       const array<double, 3>& r_EO_O,
                       array<array<array<double, 6>, 7>, 8>& Jsols,
                       const int index, const Frame Jacobian_ee) {
  // saves the two Jacobian solutions for the given joint axes at Jsols[2*index]
  // and Jsols[2*index+1]. Jacobian_ee is the frame of the Jacobian end-effector
  // ('6', '8', 'F' or 'E') r4 = r_4S_O r5 = r_5S_O
  array<double, 3> r_1ee_O, r_4ee_O, r_5ee_O;
  if (Jacobian_ee == Frame::Link6) {
    // r_P6_O = r_PS_O + r_S6_O
    //          r_PS_O - r_6S_O remember r_6S_O = r_5S_O
    r_1ee_O = {-r5[0], -r5[1], -r5[2]};
    r_4ee_O = {r4[0] - r5[0], r4[1] - r5[1], r4[2] - r5[2]};
    r_5ee_O = {0, 0, 0};
  } else if (Jacobian_ee == Frame::Flange) {
    // r_PF_O = r_PE_O + r_EF_O
    //        = r_PS_O - rEO_O + rSO_O + r_EF_O
    //        = r_PS_O - rEO_O + (0,0,d1) + 0.1034*s7_O
    r_1ee_O = {-r_EO_O[0] + 0.1034 * s7[0], -r_EO_O[1] + 0.1034 * s7[1],
               d1 - r_EO_O[2] + 0.1034 * s7[2]};
    r_4ee_O = {r4[0] - r_EO_O[0] + 0.1034 * s7[0],
               r4[1] - r_EO_O[1] + 0.1034 * s7[1],
               d1 + r4[2] - r_EO_O[2] + 0.1034 * s7[2]};
    r_5ee_O = {r5[0] - r_EO_O[0] + 0.1034 * s7[0],
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
  Jsols[2 * index][0] = {
      0,           0, 1, r_1ee_O[1],
      -r_1ee_O[0], 0};  // r_1ee_O x (0,0,1) = (r_1ee_O[1], -r_1ee_O[0], 0)
  Jsols[2 * index + 1][0] = {0, 0, 1, r_1ee_O[1], -r_1ee_O[0], 0};
  Cross_(r_1ee_O, s2, m);  // r_2ee_O = r_1ee_O
  Jsols[2 * index][1] = {s2[0], s2[1], s2[2], m[0], m[1], m[2]};
  Jsols[2 * index + 1][1] = {
      -s2[0], -s2[1], -s2[2],
      -m[0],  -m[1],  -m[2]};  // second solution of spherical shoulder
  Cross_(r_1ee_O, s3, m);      //  r3_ee = r1_ee
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
  if (Jacobian_ee == Frame::Link6) {
    Jsols[2 * index][6] = {0, 0, 0, 0, 0, 0};
    Jsols[2 * index + 1][6] = {0, 0, 0, 0, 0, 0};
  } else {
    Jsols[2 * index][6] = {s7[0], s7[1], s7[2], 0, 0, 0};
    Jsols[2 * index + 1][6] = {s7[0], s7[1], s7[2], 0, 0, 0};
  }
}

static inline array<double, 3> shoulder_axis_from_s3(
    const array<double, 3>& s3, const double horizontal_sq) {
  // Horizontal normal to the third joint axis.  One reciprocal square root
  // instead of evaluating sqrt and dividing twice.
  const double inverse = 1.0 / sqrt(horizontal_sq);
  return {-s3[1] * inverse, s3[0] * inverse, 0.0};
}

static double signed_angle(const array<double, 3>& v1,
                           const array<double, 3>& v2,
                           const array<double, 3>& s) {
  // return atan2(s[2]*(v1[0]*v2[1] - v1[1]*v2[0]) - s[1]*(v1[0]*v2[2] -
  // v1[2]*v2[0]) + s[0]*(v1[1]*v2[2] - v1[2]*v2[1]), v1[0]*v2[0] + v1[1]*v2[1]
  // + v1[2]*v2[2]);
  return nanogeofik_atan::atan2(Dot(Cross(v1, v2), s), Dot(v1, v2));
}

// The six atan2 evaluations are independent of each
// other, and nanogeofik_atan::atan2 is branch-free, so the arguments are
// gathered first and the transcendentals are evaluated in one fixed-length
// loop.  That loop vectorizes, which matters because these six calls were the
// single largest cost in every direct solver (four assembled arms per solve,
// six calls each).  The two variants that take a joint angle as given still
// evaluate all six and discard one: a vectorized six is cheaper than a scalar
// five.
static const array<double, 3> kAxisY = {0.0, 1.0, 0.0};
static const array<double, 3> kAxisZ = {0.0, 0.0, 1.0};

// Dot(Cross(v1, v2), s), written out in the same operation order.
static inline double triple_product(const array<double, 3>& v1,
                                    const array<double, 3>& v2,
                                    const array<double, 3>& s) {
  return (v1[1] * v2[2] - v2[1] * v1[2]) * s[0] +
         (v2[0] * v1[2] - v1[0] * v2[2]) * s[1] +
         (v1[0] * v2[1] - v2[0] * v1[1]) * s[2];
}

template <int kCount>
static inline void signed_angles(const double* sines, const double* cosines,
                                 double* angles) {
  double reduced[kCount];
  double polynomial[kCount];
  unsigned char octant[kCount];
  for (int i = 0; i < kCount; ++i) {
    const nanogeofik_atan::AtanReduction reduction =
        nanogeofik_atan::atan2_reduce(sines[i], cosines[i]);
    reduced[i] = reduction.u;
    octant[i] = reduction.octant;
  }
  // The one stage with no control flow, and the one that dominates the cost.
  for (int i = 0; i < kCount; ++i)
    polynomial[i] = nanogeofik_atan::atan_octant(reduced[i]);
  for (int i = 0; i < kCount; ++i)
    angles[i] =
        nanogeofik_atan::atan2_finish(polynomial[i], octant[i], sines[i]);
}

template <int N, LimitPreset Preset>
static inline void check_limits_impl(array<double, 7>& q,
                                     const JointLimits* custom_limits,
                                     bool check_bounds) {
  if constexpr (Preset == LimitPreset::None) {
    return;
  }

  const JointLimits& limits = [&]() -> const JointLimits& {
    if constexpr (Preset == LimitPreset::Panda)
      return kPandaJointLimits;
    else if constexpr (Preset == LimitPreset::FR3)
      return kFr3JointLimits;
    else if (custom_limits)
      return *custom_limits;
    else
      return kPandaJointLimits;
  }();

#pragma unroll
  for (int i = 0; i < N; i++) {
    // Joint angles reconstructed by q_from_axes come from atan2 and are
    // already in [-pi, pi]. Relative to any normal joint-limit midpoint they
    // therefore need at most one turn of wrapping. Avoid evaluating sin,
    // cos, and atan2 again for every angle in every geometric branch. Keep
    // the general identity as a cold fallback for unusually large values
    // supplied directly as the redundancy parameter (or unusual custom
    // limits), preserving the original behavior for arbitrary inputs.
    double centered;
    if constexpr (Preset == LimitPreset::Panda || Preset == LimitPreset::FR3) {
      // For Panda & FR3, joints 0, 1, 2, 4, 6 are symmetric (middle == 0.0)
      if (i != 3 && i != 5) {
        centered = q[i];
      } else {
        centered = q[i] - limits.middle[i];
      }
    } else {
      centered = q[i] - limits.middle[i];
    }

    if (centered > PI) {
      centered = centered <= 3.0 * PI
                     ? centered - TWO_PI
                     : nanogeofik_atan::atan2(sin(centered), cos(centered));
    } else if (centered < -PI) {
      centered = centered >= -3.0 * PI
                     ? centered + TWO_PI
                     : nanogeofik_atan::atan2(sin(centered), cos(centered));
    }

    if constexpr (Preset == LimitPreset::Panda || Preset == LimitPreset::FR3) {
      if (i != 3 && i != 5) {
        q[i] = centered;
      } else {
        q[i] = limits.middle[i] + centered;
      }
    } else {
      q[i] = limits.middle[i] + centered;
    }

    if (check_bounds) {
      if (q[i] < limits.lower[i])
        q[i] =
            q[i] >= limits.lower[i] - JOINT_LIMIT_TOL ? limits.lower[i] : NAN;
      else if (q[i] > limits.upper[i])
        q[i] =
            q[i] <= limits.upper[i] + JOINT_LIMIT_TOL ? limits.upper[i] : NAN;
    }
  }
}

template <int N>
static inline void check_limits(array<double, 7>& q,
                                const SolverTuning& tuning) {
  if (tuning.limit_preset == LimitPreset::None) {
    return;
  }

  const bool check_bounds = tuning.check_joint_limits;
  if (tuning.custom_limits != nullptr) {
    check_limits_impl<N, LimitPreset::Custom>(q, tuning.custom_limits,
                                              check_bounds);
    return;
  }

  switch (tuning.limit_preset) {
    case LimitPreset::Panda:
      check_limits_impl<N, LimitPreset::Panda>(q, nullptr, check_bounds);
      break;
    case LimitPreset::FR3:
      check_limits_impl<N, LimitPreset::FR3>(q, nullptr, check_bounds);
      break;
    case LimitPreset::Custom:
      check_limits_impl<N, LimitPreset::Custom>(q, nullptr, check_bounds);
      break;
    case LimitPreset::None:
      break;
  }
}

static inline void check_limits(array<double, 7>& q, int n,
                                const SolverTuning& tuning) {
  if (n == 7) {
    check_limits<7>(q, tuning);
  } else if (n == 3) {
    check_limits<3>(q, tuning);
  } else {
    check_limits<7>(q, tuning);
  }
}

static inline void check_limits(array<double, 7>& q, int n) {
  SolverTuning tuning;
  check_limits(q, n, tuning);
}

// Recover the joint angles directly from the solved screw-axis directions.
//
// Algorithm 1 in the paper rotates all remaining home axes after recovering
// each angle.  For the Franka's home-axis sequence
//
//   s1=z, s2=y, s3=z, s4=-y, s5=z, s6=-y, s7=-z,
//
// those propagated reference axes are already available from neighboring
// solved axes: y, z, -s2, s3, s4, and -s5 respectively.  Using them directly
// is algebraically identical, but removes the temporary Eigen Jacobian and 20
// vector rotations from every geometric branch.
static array<double, 6> q_from_axes(const array<double, 3>& s2,
                                    const array<double, 3>& s3,
                                    const array<double, 3>& s4,
                                    const array<double, 3>& s5,
                                    const array<double, 3>& s6,
                                    const array<double, 3>& s7) {
  const array<double, 3> minus_s2 = {-s2[0], -s2[1], -s2[2]};
  const array<double, 3> minus_s5 = {-s5[0], -s5[1], -s5[2]};
  const double sines[6] = {
      triple_product(kAxisY, s2, kAxisZ), triple_product(kAxisZ, s3, s2),
      triple_product(minus_s2, s4, s3),   triple_product(s3, s5, s4),
      triple_product(s4, s6, s5),         triple_product(minus_s5, s7, s6)};
  const double cosines[6] = {Dot(kAxisY, s2),   Dot(kAxisZ, s3),
                             Dot(minus_s2, s4), Dot(s3, s5),
                             Dot(s4, s6),       Dot(minus_s5, s7)};
  array<double, 6> q;
  signed_angles<6>(sines, cosines, q.data());
  return q;
}

static array<double, 6> q_from_axes_with_q4(
    const array<double, 3>& s2, const array<double, 3>& s3,
    const array<double, 3>& s4, const array<double, 3>& s5,
    const array<double, 3>& s6, const array<double, 3>& s7, const double q4) {
  const array<double, 3> minus_s2 = {-s2[0], -s2[1], -s2[2]};
  const array<double, 3> minus_s5 = {-s5[0], -s5[1], -s5[2]};
  const double sines[5] = {
      triple_product(kAxisY, s2, kAxisZ), triple_product(kAxisZ, s3, s2),
      triple_product(minus_s2, s4, s3), triple_product(s4, s6, s5),
      triple_product(minus_s5, s7, s6)};
  const double cosines[5] = {Dot(kAxisY, s2), Dot(kAxisZ, s3),
                             Dot(minus_s2, s4), Dot(s4, s6), Dot(minus_s5, s7)};
  double angles[5];
  signed_angles<5>(sines, cosines, angles);
  return {angles[0], angles[1], angles[2], q4, angles[3], angles[4]};
}

static array<double, 6> q_from_axes_with_q6(const array<double, 3>& s2,
                                            const array<double, 3>& s3,
                                            const array<double, 3>& s4,
                                            const array<double, 3>& s5,
                                            const array<double, 3>& s6,
                                            const double q6) {
  const array<double, 3> minus_s2 = {-s2[0], -s2[1], -s2[2]};
  const double sines[5] = {
      triple_product(kAxisY, s2, kAxisZ), triple_product(kAxisZ, s3, s2),
      triple_product(minus_s2, s4, s3), triple_product(s3, s5, s4),
      triple_product(s4, s6, s5)};
  const double cosines[5] = {Dot(kAxisY, s2), Dot(kAxisZ, s3),
                             Dot(minus_s2, s4), Dot(s3, s5), Dot(s4, s6)};
  array<double, 6> q;
  signed_angles<5>(sines, cosines, q.data());
  q[5] = q6;
  return q;
}

static array<double, 6> q_from_axes_with_q5(
    const array<double, 3>& s2, const array<double, 3>& s3,
    const array<double, 3>& s4, const array<double, 3>& s5,
    const array<double, 3>& s6, const array<double, 3>& s7, const double q5) {
  const array<double, 3> minus_s2 = {-s2[0], -s2[1], -s2[2]};
  const array<double, 3> minus_s5 = {-s5[0], -s5[1], -s5[2]};
  const double sines[5] = {
      triple_product(kAxisY, s2, kAxisZ), triple_product(kAxisZ, s3, s2),
      triple_product(minus_s2, s4, s3), triple_product(s3, s5, s4),
      triple_product(minus_s5, s7, s6)};
  const double cosines[5] = {Dot(kAxisY, s2), Dot(kAxisZ, s3),
                             Dot(minus_s2, s4), Dot(s3, s5), Dot(minus_s5, s7)};
  array<double, 6> q;
  signed_angles<5>(sines, cosines, q.data());
  // Computed angles are q1..q4 and q6; the locked value takes index 4.
  return {q[0], q[1], q[2], q[3], q5, q[4]};
}

static double opposite_principal_angle(const double angle) {
  return angle > 0.0 ? angle - PI : angle + PI;
}

static array<double, 3> q_from_flipped_shoulder(const array<double, 6>& q) {
  // The second spherical-shoulder assembly negates s2.  In Algorithm 1 this
  // negates q2's rotation axis and reverses the reference directions for q1
  // and q3.  Consequently its first three angles follow directly from the
  // already recovered branch: opposite(q1), -q2, opposite(q3).  Re-running
  // three signed-angle atan2 evaluations is redundant.
  return {opposite_principal_angle(q[0]), -q[1],
          opposite_principal_angle(q[2])};
}

// whole Jacobian and q
array<double, 7> J_to_q(const array<array<double, 6>, 7>& J,
                        const array<array<double, 3>, 3>& R, const Frame ee) {
  // J is the transpose of the Jacobian
  // R is the rotation matrix of frame ee
  // ee must be a frame attached to the gripper: "E", "F" or "8".
  array<double, 7> q;
  array<double, 3> i7, ie, s2, s3, s4, s5, s6, s7;
  s2 = {J[1][0], J[1][1], J[1][2]};
  s3 = {J[2][0], J[2][1], J[2][2]};
  s4 = {J[3][0], J[3][1], J[3][2]};
  s5 = {J[4][0], J[4][1], J[4][2]};
  s6 = {J[5][0], J[5][1], J[5][2]};
  s7 = {J[6][0], J[6][1], J[6][2]};
  Cross_(s6, s7, i7);
  ie = {R[0][0], R[1][0], R[2][0]};
  q[6] = signed_angle(i7, ie, s7) + (ee == Frame::EndEffector ? PI / 4 : 0);
  const array<double, 6> lower = q_from_axes(s2, s3, s4, s5, s6, s7);
  std::copy(lower.begin(), lower.end(), q.begin());
  return q;
}

static void column_1s_times_vec(const array<double, 9>& R,
                                const array<double, 3>& v,
                                array<double, 3>& res) {
  res[0] = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
  res[1] = R[3] * v[0] + R[4] * v[1] + R[5] * v[2];
  res[2] = R[6] * v[0] + R[7] * v[1] + R[8] * v[2];
}

// Frame E is orthonormal, so rotating its i axis about k by
// delta = pi/4-q7 stays exactly in the i/j plane.  The associated sixth joint
// axis is k x i6.  Constructing both together avoids a general Rodrigues
// rotation followed by a cross product.
static void wrist_axes_from_q7(const array<double, 3>& i_E_O,
                               const array<double, 3>& j_E_O, const double q7,
                               array<double, 3>& i_6_O, array<double, 3>& s6) {
  const double delta = PI / 4.0 - q7;
  const double c = cos(delta);
  const double s = sin(delta);
  for (int component = 0; component < 3; ++component) {
    i_6_O[component] = c * i_E_O[component] + s * j_E_O[component];
    s6[component] = c * j_E_O[component] - s * i_E_O[component];
  }
}

static inline void rotate_by_sin_cos(const array<double, 3>& s, const double st,
                                     const double ct, const array<double, 3>& v,
                                     array<double, 3>& res) {
  const double dot = Dot(s, v);
  const double one_minus_ct_dot = (1.0 - ct) * dot;
  res = {
      ct * v[0] + st * (s[1] * v[2] - s[2] * v[1]) + one_minus_ct_dot * s[0],
      ct * v[1] + st * (s[2] * v[0] - s[0] * v[2]) + one_minus_ct_dot * s[1],
      ct * v[2] + st * (s[0] * v[1] - s[1] * v[0]) + one_minus_ct_dot * s[2]};
}

// Rotate v about the (unit) axis s by beta1 and scale by 1/b1, which is what
// every caller of the beta1 rotation actually wants: the unit third joint axis.
static inline void rotate_by_beta1_scaled(const array<double, 3>& s,
                                          const array<double, 3>& v,
                                          array<double, 3>& res) {
  const double dot = Dot(s, v);
  const double one_minus_ct_dot = (1.0 - cos_beta1) * dot;
  res = {(cos_beta1 * v[0] + sin_beta1 * (s[1] * v[2] - s[2] * v[1]) +
          one_minus_ct_dot * s[0]) *
             inv_b1,
         (cos_beta1 * v[1] + sin_beta1 * (s[2] * v[0] - s[0] * v[2]) +
          one_minus_ct_dot * s[1]) *
             inv_b1,
         (cos_beta1 * v[2] + sin_beta1 * (s[0] * v[1] - s[1] * v[0]) +
          one_minus_ct_dot * s[2]) *
             inv_b1};
}

static void rotate_by_axis_angle(const array<double, 3>& s, const double theta,
                                 const array<double, 3>& v,
                                 array<double, 3>& res) {
  rotate_by_sin_cos(s, sin(theta), cos(theta), v, res);
}

static Eigen::Matrix4d T_rpy(const double r, const double p, const double y,
                             const double px, const double py,
                             const double pz) {
  Eigen::Matrix4d T;
  T << cos(p) * cos(y), cos(y) * sin(p) * sin(r) - cos(r) * sin(y),
      sin(r) * sin(y) + cos(r) * cos(y) * sin(p), px, cos(p) * sin(y),
      cos(r) * cos(y) + sin(p) * sin(r) * sin(y),
      cos(r) * sin(p) * sin(y) - cos(y) * sin(r), py, -sin(p), cos(p) * sin(r),
      cos(p) * cos(r), pz, 0, 0, 0, 1;
  return T;
}

static Eigen::Matrix4d T_rot_z(const double theta, const double px = 0.0,
                               const double py = 0.0, const double pz = 0.0) {
  Eigen::Matrix4d T;
  T << cos(theta), -sin(theta), 0, px, sin(theta), cos(theta), 0, py, 0, 0, 1,
      pz, 0, 0, 0, 1;
  return T;
}

static void get_frame_transforms(array<Eigen::Matrix4d, 9>& Ti,
                                 const array<double, 7>& q) {
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

static unsigned int ee_number(const Frame ee) {
  switch (ee) {
    case Frame::EndEffector:
      return 9;
    case Frame::Flange:
      return 8;
    case Frame::Link7:
      return 7;
    case Frame::Link6:
      return 6;
    case Frame::Link5:
      return 5;
    case Frame::Link4:
      return 4;
    case Frame::Link3:
      return 3;
    case Frame::Link2:
      return 2;
    case Frame::Link1:
      return 1;
  }
  return 9;
}

static void post_rotate_z(Eigen::Matrix3d& rotation, const double angle);
static void post_rotate_quarter_x_then_z(Eigen::Matrix3d& rotation,
                                         const double angle, const int sign);
array<array<double, 6>, 7> J_from_q(const array<double, 7>& q, const Frame ee) {
  // returns J^T for a given vector of joint angles, q. The end-effector frame
  // is ee OUTPUT: J^T \in R^(7,6): array<array<double,6>,7> INPUT: q \in R^7:
  // array<double,7>
  //        ee
  //
  // The chain is propagated by rotating the three
  // rotation columns directly (the same scheme as franka_flange_fk) instead of
  // building nine homogeneous matrices and multiplying them sequentially; the
  // screw axes and origins are snapshotted as each joint frame is reached. The
  // adjoint transform that shifts the linear rows to the end-effector origin
  // is applied inline: v' = v + (-p_ee) x omega. Same algebra, no 4x4
  // products.
  const unsigned int een = ee_number(ee);
  const unsigned int cols = een >= 7 ? 7 : een;

  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d pos(0.0, 0.0, d1);
  Eigen::Matrix<double, 3, 7> axes;
  Eigen::Matrix<double, 3, 7> origins;
  auto snapshot = [&](unsigned int j) {
    axes.col(j) = R.col(2);
    origins.col(j) = pos;
  };

  post_rotate_z(R, q[0]);  // segment 0 -> frame 1
  snapshot(0);
  post_rotate_quarter_x_then_z(R, q[1], -1);  // segment 1 -> frame 2
  snapshot(1);
  pos -= d3 * R.col(1);
  post_rotate_quarter_x_then_z(R, q[2], 1);  // segment 2 -> frame 3
  snapshot(2);
  pos += a4 * R.col(0);
  post_rotate_quarter_x_then_z(R, q[3], 1);  // segment 3 -> frame 4
  snapshot(3);
  pos += -a5 * R.col(0) + d5 * R.col(1);
  post_rotate_quarter_x_then_z(R, q[4], -1);  // segment 4 -> frame 5
  snapshot(4);
  post_rotate_quarter_x_then_z(R, q[5], 1);  // segment 5 -> frame 6
  snapshot(5);
  pos += a7 * R.col(0);
  post_rotate_quarter_x_then_z(R, q[6], 1);  // segment 6 -> frame 7
  snapshot(6);
  if (een >= 8) pos += 0.107 * R.col(2);  // segment 7 (frame 8 / F)
  if (een >= 9) {                         // segment 8 (E)
    pos += 0.1034 * R.col(2);
    post_rotate_z(R, -PI / 4);
  }

  array<array<double, 6>, 7> Jarr;
  for (unsigned int i = 0; i < cols; i++) {
    const Eigen::Vector3d s = axes.col(i);
    const Eigen::Vector3d m = origins.col(i).cross(s) + (-pos).cross(s);
    Jarr[i] = {s[0], s[1], s[2], m[0], m[1], m[2]};
  }
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

static void post_rotate_quarter_x_then_z(Eigen::Matrix3d& rotation,
                                         const double angle, const int sign) {
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
  // x rotations. Propagating the three rotation columns directly avoids
  // creating nine 4x4 matrices and performing seven general homogeneous
  // multiplications.
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

Eigen::Matrix4d franka_fk(const array<double, 7>& q, const Frame ee) {
  // Forward kinematics function
  // INPUT: joint angles q, and end effector name ee
  // OUTPUT: TOee is the transformation matrix of frame ee w.r.t. frame O
  if (ee == Frame::Flange) return franka_flange_fk(q);
  if (ee == Frame::EndEffector) {
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

static void franka_fk_all_frames(array<Eigen::Matrix4d, 9>& Ts,
                                 const array<double, 7>& q) {
  // Forward kinematics function saving in Ts the transformation matrices of all
  // frames w.r.t. frame O
  array<Eigen::Matrix4d, 9> Ti;
  get_frame_transforms(Ti, q);
  Ts[0] = Ti[0];
  for (int i = 1; i < 9; i++) Ts[i] = Ts[i - 1] * Ti[i];
}

NANOGEOFIK_TARGET_CLONES
unsigned int franka_ik_q7(const array<double, 3>& r,
                          const array<double, 9>& ROE, const double q7,
                          array<array<double, 7>, 8>& qsols,
                          const SolverTuning& tuning) {
  const double q1_sing = tuning.q1_sing;
  // IK with q7 as free variable
  // INPUT: r = r_EO_O, position of frame E in frame O
  //        ROE, orientation of frame E in frame O
  //        q7, joint angle of joint 7
  //        qsols, array to store 8 solutions
  //        q1_sing, emergency value of q1 in case of singularity at shoulder
  //        joints.
  // OUTPUT: number of solutions found.
  // ri = r_iS_O, i = 1,2,3,4,5,6,7
  // si = s_i_O
  const array<double, 3> i_E_O = {ROE[0], ROE[3], ROE[6]};
  const array<double, 3> j_E_O = {ROE[1], ROE[4], ROE[7]};
  const array<double, 3> k_E_O = {ROE[2], ROE[5], ROE[8]};
  array<double, 3> i_6_O;
  array<double, 3> s6;
  wrist_axes_from_q7(i_E_O, j_E_O, q7, i_6_O, s6);
  array<double, 3> r6 = {r[0] - dE * k_E_O[0] - a7 * i_6_O[0],
                         r[1] - dE * k_E_O[1] - a7 * i_6_O[1],
                         r[2] - d1 - dE * k_E_O[2] - a7 * i_6_O[2]};
  double l = Norm(r6);
  double tmp = (b1 * b1 - l * l - b2 * b2) / (-2 * l * b2);
  if (!clamp_trig_roundoff(tmp)) {
    nanogeofik_log() << "ERROR: unable to assembly kinematic chain";
    for (auto& solution : qsols) fill(solution.begin(), solution.end(), NAN);
    return 0;
  }
  const double cos_actmp = tmp;
  const double sin_actmp = sqrt(std::max(0.0, 1.0 - cos_actmp * cos_actmp));
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
  array<double, 4> inverse_s4_norms;
  double sa2, ca2;
  // The elbow-down assembly (branch 1) is generated only inside the flat
  // window d3+d5 < l < b1+b2.  That condition is algebraically identical to
  // sin(alpha2) > 0 on this branch: at l = d3+d5 the triangle angle at O6
  // equals beta2 exactly, and with a4 == a5 the equivalence is exact.  Its
  // purpose is therefore not collision filtering -- on the Panda, link
  // interference is encoded in the joint limits themselves (joint 4 may
  // not cross zero), and every elbow-down assembly outside the window has
  // q4 >= 0, i.e. violates the limits and would be rejected by
  // check_limits() anyway (verified structurally over 2e6 samples).  It
  // also happens to be the region where sin(alpha2) < 0, where a
  // |sa2|-normalized s4 would not close the chain.
  int n_alphs = 1;
  unsigned int n_sols = 0;
  if (d3 + d5 < l && l < b1 + b2) n_alphs = 2;
  double v[3];
  for (int i = 0; i < n_alphs; i++) {
    // Upstream advanced alpha2 to the second elbow
    // assembly at the *end* of the loop body, where the `continue` below skips
    // past it -- so a first branch that bailed left the second iteration
    // re-solving the identical branch instead of the elbow flip. Selecting from
    // the index makes the two iterations independent.
    const double branch_sin_actmp = i == 0 ? sin_actmp : -sin_actmp;
    sa2 = sin_beta2 * cos_actmp + cos_beta2 * branch_sin_actmp;
    ca2 = cos_beta2 * cos_actmp - sin_beta2 * branch_sin_actmp;
    tmp = -rz * ca2 / (ry * sa2);
    if (!clamp_trig_roundoff(tmp)) continue;
    const double sin_gamma = tmp;
    const double cos_gamma = sqrt(std::max(0.0, 1.0 - sin_gamma * sin_gamma));
    v[0] = -sa2 * cos_gamma;
    v[1] = -sa2 * sin_gamma;
    v[2] = -ca2;
    s5s[n_sols] = {i_C_O[0] * v[0] + j_C_O[0] * v[1] + k_C_O[0] * v[2],
                   i_C_O[1] * v[0] + j_C_O[1] * v[1] + k_C_O[1] * v[2],
                   i_C_O[2] * v[0] + j_C_O[2] * v[1] + k_C_O[2] * v[2]};
    tmp = 2 * sa2 * cos_gamma;
    // s5[n_sols+1] = s5s[n_sols] + (2*sa2*cos(tmp)*i_C_O);
    s5s[n_sols + 1] = {s5s[n_sols][0] + tmp * i_C_O[0],
                       s5s[n_sols][1] + tmp * i_C_O[1],
                       s5s[n_sols][2] + tmp * i_C_O[2]};
    inverse_s4_norms[n_sols] = inverse_s4_norms[n_sols + 1] =
        1.0 / (l * fabs(sa2));
    n_sols += 2;
  }
  array<double, 6> sol1;
  array<double, 3> sol2;
  array<double, 3> s4, r4, s3, s2, s5;
  for (unsigned int i = 0; i < n_sols; i++) {
    s5 = s5s[i];
    Cross_(s5, r6, s4);
    s4 = {s4[0] * inverse_s4_norms[i], s4[1] * inverse_s4_norms[i],
          s4[2] * inverse_s4_norms[i]};
    Cross_(s5, s4, r4);
    r4 = {r6[0] - d5 * s5[0] + a5 * r4[0], r6[1] - d5 * s5[1] + a5 * r4[1],
          r6[2] - d5 * s5[2] + a5 * r4[2]};
    rotate_by_beta1_scaled(s4, r4, s3);
    tmp = s3[1] * s3[1] + s3[0] * s3[0];
    if (tmp > SHOULDER_SING_TOL * SHOULDER_SING_TOL) {
      s2 = shoulder_axis_from_s3(s3, tmp);
    } else {
      s2 = {sin(q1_sing), cos(q1_sing), 0};
    }
    sol1 = q_from_axes(s2, s3, s4, s5, s6, k_E_O);
    sol2 = q_from_flipped_shoulder(sol1);
    qsols[2 * i] = {sol1[0], sol1[1], sol1[2], sol1[3], sol1[4], sol1[5], q7};
    check_limits(qsols[2 * i], 7, tuning);
    qsols[2 * i + 1] = {sol2[0],         sol2[1],         sol2[2],
                        qsols[2 * i][3], qsols[2 * i][4], qsols[2 * i][5],
                        qsols[2 * i][6]};
    check_limits(qsols[2 * i + 1], 3, tuning);
  }
  for (int i = 2 * n_sols; i < 8; i++)
    fill(qsols[i].begin(), qsols[i].end(), NAN);
  return 2 * n_sols;
}

NANOGEOFIK_TARGET_CLONES
unsigned int franka_ik_q4(const array<double, 3>& r,
                          const array<double, 9>& ROE, const double q4,
                          array<array<double, 7>, 8>& qsols,
                          const SolverTuning& tuning) {
  const double q1_sing = tuning.q1_sing;
  const double q7_sing = tuning.q7_sing;
  // IK with q4 as free variable
  // INPUT: r = r_EO_O, position of frame E in frame O
  //        ROE, orientation of frame E in frame O (row-first format)
  //        q4, joint angle of joint 4
  //        qsols, array to store 8 solutions
  //        q1_sing, emergency value of q1 in case of singularity at shoulder
  //        joints (type-1 singularity). q7_sing, emergency value of q7 in case
  //        of singularity of S7 intersecting S (type-2 singularity).
  // OUTPUT: number of solutions found.
  // ri = r_iS_O, i = 1,2,3,4,5,6,7
  // si = s_i_O
  array<double, 3> r_ES_O = {r[0], r[1], r[2] - d1};
  array<double, 3> tmp_v = {r_ES_O[1] * ROE[8] - r_ES_O[2] * ROE[5],
                            r_ES_O[2] * ROE[2] - r_ES_O[0] * ROE[8],
                            r_ES_O[0] * ROE[5] - r_ES_O[1] * ROE[2]};
  if (tmp_v[0] * tmp_v[0] + tmp_v[1] * tmp_v[1] + tmp_v[2] * tmp_v[2] <
      SING_TOL * SING_TOL)
    return franka_ik_q7(r, ROE, q7_sing, qsols, SolverTuning{q1_sing});
  array<double, 3> r_O7S_O = {r_ES_O[0] - dE * ROE[2], r_ES_O[1] - dE * ROE[5],
                              r_ES_O[2] - dE * ROE[8]};
  array<double, 3> r_O7S_E = {
      ROE[0] * r_O7S_O[0] + ROE[3] * r_O7S_O[1] + ROE[6] * r_O7S_O[2],
      ROE[1] * r_O7S_O[0] + ROE[4] * r_O7S_O[1] + ROE[7] * r_O7S_O[2],
      ROE[2] * r_O7S_O[0] + ROE[5] * r_O7S_O[1] + ROE[8] * r_O7S_O[2]};
  const double alpha = q4 + beta1 + beta2 - PI;
  const double sin_alpha = sin(alpha);
  const double cos_alpha = cos(alpha);
  double lo2 = b1 * b1 + b2 * b2 - 2 * b1 * b2 * cos_alpha;
  double lp2 = lo2 - r_O7S_E[2] * r_O7S_E[2];
  // fixed triangle condition
  if (lp2 < 0) {
    if (lp2 * lp2 < SING_TOL * SING_TOL)
      lp2 = 0;
    else {
      nanogeofik_log() << "\nERROR: unable to assembly kinematic chain\n";
      for (int i = 0; i < 8; ++i) fill(qsols[i].begin(), qsols[i].end(), NAN);
      return 0;
    }
  }
  double sin_gamma_offset = b1 * sin_alpha / sqrt(lo2);
  if (!clamp_trig_roundoff(sin_gamma_offset)) {
    for (auto& solution : qsols) fill(solution.begin(), solution.end(), NAN);
    return 0;
  }
  const double cos_gamma_offset =
      sqrt(std::max(0.0, 1.0 - sin_gamma_offset * sin_gamma_offset));
  const double cg2 =
      cos_beta2 * cos_gamma_offset - sin_beta2 * sin_gamma_offset;
  const double sg2 =
      sin_beta2 * cos_gamma_offset + cos_beta2 * sin_gamma_offset;
  const double Lp2 = r_O7S_E[0] * r_O7S_E[0] + r_O7S_E[1] * r_O7S_E[1];
  const double sqrt_Lp2 = sqrt(Lp2);
  const double phi = nanogeofik_atan::atan2(-r_O7S_E[1], -r_O7S_E[0]);
  double tmp = (Lp2 + a7 * a7 - lp2) / (2 * sqrt_Lp2 * a7);
  // fixed condition
  if (tmp > 1.0) {
    if ((tmp - 1) * (tmp - 1) < SING_TOL * SING_TOL)
      tmp = 1.0;
    else {
      nanogeofik_log() << "\nERROR: unable to assembly kinematic chain\n";
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
      nanogeofik_log() << "\nERROR: unable to assembly kinematic chain\n";
      for (int i = 0; i < 8; ++i) {
          fill(qsols[i].begin(), qsols[i].end(), NAN);
      }
      return 0;
  }
  */
  const double psi = acos(tmp);
  const double sin_psi = sqrt(std::max(0.0, 1.0 - tmp * tmp));
  const double cos_phi = -r_O7S_E[0] / sqrt_Lp2;
  const double sin_phi = -r_O7S_E[1] / sqrt_Lp2;
  const double wrist_cosines[2] = {sin_phi * tmp + cos_phi * sin_psi,
                                   sin_phi * tmp - cos_phi * sin_psi};
  const double wrist_sines[2] = {-(cos_phi * tmp - sin_phi * sin_psi),
                                 -(cos_phi * tmp + sin_phi * sin_psi)};
  double ry, rz;
  double q7s[2] = {-phi - psi - 3 * PI / 4, -phi + psi - 3 * PI / 4};
  unsigned int ind = 0;
  array<double, 3> s2, s3, s4, s5, s6, r4, r6, i_C_O, j_C_O, k_C_O;
  array<double, 6> sol1;
  array<double, 3> sol2;
  for (unsigned int q7_index = 0; q7_index < 2; ++q7_index) {
    const double q7 = q7s[q7_index];
    const double wrist_cos = wrist_cosines[q7_index];
    const double wrist_sin = wrist_sines[q7_index];
    tmp_v = {wrist_cos, wrist_sin, 0};
    s6 = {ROE[0] * tmp_v[0] + ROE[1] * tmp_v[1],
          ROE[3] * tmp_v[0] + ROE[4] * tmp_v[1],
          ROE[6] * tmp_v[0] + ROE[7] * tmp_v[1]};
    tmp_v = {-a7 * wrist_sin, a7 * wrist_cos, 0};
    r6 = {ROE[0] * tmp_v[0] + ROE[1] * tmp_v[1],
          ROE[3] * tmp_v[0] + ROE[4] * tmp_v[1],
          ROE[6] * tmp_v[0] + ROE[7] * tmp_v[1]};
    r6 = {r6[0] + r_O7S_O[0], r6[1] + r_O7S_O[1], r6[2] + r_O7S_O[2]};
    const double r6_norm = Norm(r6);
    k_C_O = {-r6[0] / r6_norm, -r6[1] / r6_norm, -r6[2] / r6_norm};
    Cross_(k_C_O, s6, i_C_O);
    tmp = Norm(i_C_O);
    i_C_O = {i_C_O[0] / tmp, i_C_O[1] / tmp, i_C_O[2] / tmp};
    Cross_(k_C_O, i_C_O, j_C_O);
    ry = s6[0] * j_C_O[0] + s6[1] * j_C_O[1] + s6[2] * j_C_O[2];
    rz = s6[0] * k_C_O[0] + s6[1] * k_C_O[1] + s6[2] * k_C_O[2];
    tmp = -rz * cg2 / (ry * sg2);
    if (tmp * tmp > 1) continue;
    const double sin_gamma = tmp;
    const double cos_gamma = sqrt(std::max(0.0, 1.0 - sin_gamma * sin_gamma));
    for (int gamma_branch = 0; gamma_branch < 2; ++gamma_branch) {
      tmp_v = {(gamma_branch == 0 ? -sg2 : sg2) * cos_gamma, -sg2 * sin_gamma,
               -cg2};
      s5 = {i_C_O[0] * tmp_v[0] + j_C_O[0] * tmp_v[1] + k_C_O[0] * tmp_v[2],
            i_C_O[1] * tmp_v[0] + j_C_O[1] * tmp_v[1] + k_C_O[1] * tmp_v[2],
            i_C_O[2] * tmp_v[0] + j_C_O[2] * tmp_v[1] + k_C_O[2] * tmp_v[2]};
      Cross_(s5, r6, s4);
      const double inverse_s4_norm = 1.0 / (r6_norm * fabs(sg2));
      s4 = {s4[0] * inverse_s4_norm, s4[1] * inverse_s4_norm,
            s4[2] * inverse_s4_norm};
      Cross_(s5, s4, r4);
      r4 = {r6[0] - d5 * s5[0] + a5 * r4[0], r6[1] - d5 * s5[1] + a5 * r4[1],
            r6[2] - d5 * s5[2] + a5 * r4[2]};
      rotate_by_beta1_scaled(s4, r4, s3);
      tmp = s3[1] * s3[1] + s3[0] * s3[0];
      if (tmp > SHOULDER_SING_TOL * SHOULDER_SING_TOL)
        s2 = shoulder_axis_from_s3(s3, tmp);
      else
        s2 = {sin(q1_sing), cos(q1_sing), 0};
      sol1 = q_from_axes_with_q4(s2, s3, s4, s5, s6,
                                 array<double, 3>{ROE[2], ROE[5], ROE[8]}, q4);
      sol2 = q_from_flipped_shoulder(sol1);
      qsols[2 * ind] = {sol1[0], sol1[1], sol1[2], sol1[3],
                        sol1[4], sol1[5], q7};
      check_limits(qsols[2 * ind], 7, tuning);
      qsols[2 * ind + 1] = {sol2[0],           sol2[1],
                            sol2[2],           qsols[2 * ind][3],
                            qsols[2 * ind][4], qsols[2 * ind][5],
                            qsols[2 * ind][6]};
      check_limits(qsols[2 * ind + 1], 3, tuning);
      ind++;
    }
  }
  for (int i = 2 * ind; i < 8; ++i) {
    fill(qsols[i].begin(), qsols[i].end(), NAN);
  }
  return 2 * ind;
}

static unsigned int franka_ik_q6_parallel(const array<double, 3>& r_ES_O,
                                          const array<double, 9>& ROE,
                                          const int sgn,
                                          array<array<double, 7>, 8>& qsols,
                                          const SolverTuning& tuning) {
  const double q1_sing = tuning.q1_sing;
  // Parallel case of the IK with q6 as free variable. Only called by
  // franka_ik_q6(), not by the user. INPUT: r_ES_O, ROE, sgn  = sign(cos(q6)),
  // qsols, q1_sing. OUTPUT: number of solutions found. NOTATION: ri = r_iS_O, i
  // = 1,2,3,4,5,6,7 si = s_i_O Q is a frame that is parallel to frame E and has
  // origin at Q
  array<double, 3> s7 = {ROE[2], ROE[5], ROE[8]};
  array<double, 3> r_QS_O = {r_ES_O[0] + (-dE + sgn * d5) * s7[0],
                             r_ES_O[1] + (-dE + sgn * d5) * s7[1],
                             r_ES_O[2] + (-dE + sgn * d5) * s7[2]};
  array<double, 3> r_SQ_Q = {
      -ROE[0] * r_QS_O[0] - ROE[3] * r_QS_O[1] - ROE[6] * r_QS_O[2],
      -ROE[1] * r_QS_O[0] - ROE[4] * r_QS_O[1] - ROE[7] * r_QS_O[2],
      -ROE[2] * r_QS_O[0] - ROE[5] * r_QS_O[1] - ROE[8] * r_QS_O[2]};
  double tmp = b1 * b1 - r_SQ_Q[2] * r_SQ_Q[2];
  if (tmp * tmp < SING_TOL * SING_TOL) tmp = 0;
  if (tmp < 0) {
    nanogeofik_log() << "ERROR: unable to assembly kinematic chain";
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
  array<double, 3> tmp_v, r_O6pQ_Q, i_4_Q, r_O4Q_Q, s6_Q, r_O6_Q, s4_Q, s3_Q,
      s2, s3, s4, s5, s6;
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
      r_O6pQ_Q = {a7 * r_O6pQ_Q[0] / l_SpQ, a7 * r_O6pQ_Q[1] / l_SpQ,
                  a7 * r_O6pQ_Q[2] / l_SpQ};
      i_4_Q = {r_SpQ_Q[0] - r_O6pQ_Q[0], r_SpQ_Q[1] - r_O6pQ_Q[1],
               r_SpQ_Q[2] - r_O6pQ_Q[2]};
      tmp = Norm(i_4_Q);
      tmp_sgn = L < 0 ? -1 : 1;
      i_4_Q = {tmp_sgn * i_4_Q[0] / tmp, tmp_sgn * i_4_Q[1] / tmp,
               tmp_sgn * i_4_Q[2] / tmp};
      r_O4Q_Q = {r_O6pQ_Q[0] + a5 * i_4_Q[0], r_O6pQ_Q[1] + a5 * i_4_Q[1],
                 r_O6pQ_Q[2] + a5 * i_4_Q[2]};
      Cross_(r_O6pQ_Q, k, s6_Q);
      s6_Q = {s6_Q[0] / a7, s6_Q[1] / a7, s6_Q[2] / a7};
      r_O6_Q = {r_O6pQ_Q[0], r_O6pQ_Q[1], r_O6pQ_Q[2] - sgn * d5};
      Cross_(i_4_Q, s5_Q, s4_Q);
      tmp_v = {r_O4Q_Q[0] - r_SQ_Q[0], r_O4Q_Q[1] - r_SQ_Q[1],
               r_O4Q_Q[2] - r_SQ_Q[2]};
      rotate_by_sin_cos(s4_Q, sin_beta1, cos_beta1, tmp_v, s3_Q);
      tmp = Norm(s3_Q);
      partial_J_Q << s3_Q[0] / tmp, s4_Q[0], s5_Q[0], s6_Q[0], s3_Q[1] / tmp,
          s4_Q[1], s5_Q[1], s6_Q[1], s3_Q[2] / tmp, s4_Q[2], s5_Q[2], s6_Q[2];
      partial_J_O = ROQ * partial_J_Q;
      s3 = {partial_J_O(0, 0), partial_J_O(1, 0), partial_J_O(2, 0)};
      s4 = {partial_J_O(0, 1), partial_J_O(1, 1), partial_J_O(2, 1)};
      s5 = {partial_J_O(0, 2), partial_J_O(1, 2), partial_J_O(2, 2)};
      s6 = {partial_J_O(0, 3), partial_J_O(1, 3), partial_J_O(2, 3)};
      q7 = nanogeofik_atan::atan2(r_O6pQ_Q[1], -r_O6pQ_Q[0]) + PI / 4;
      tmp = s3[1] * s3[1] + s3[0] * s3[0];
      if (tmp > SHOULDER_SING_TOL * SHOULDER_SING_TOL)
        s2 = shoulder_axis_from_s3(s3, tmp);
      else
        s2 = {sin(q1_sing), cos(q1_sing), 0};
      sol1 = q_from_axes(s2, s3, s4, s5, s6, s7);
      sol2 = q_from_flipped_shoulder(sol1);
      qsols[2 * ind] = {sol1[0], sol1[1], sol1[2], sol1[3],
                        sol1[4], sol1[5], q7};
      check_limits(qsols[2 * ind], 7, tuning);
      qsols[2 * ind + 1] = {sol2[0],           sol2[1],
                            sol2[2],           qsols[2 * ind][3],
                            qsols[2 * ind][4], qsols[2 * ind][5],
                            qsols[2 * ind][6]};
      check_limits(qsols[2 * ind + 1], 3, tuning);
      ind++;
    }
  }
  for (int i = 2 * ind; i < 8; ++i) {
    fill(qsols[i].begin(), qsols[i].end(), NAN);
  }
  return 2 * ind;
}

NANOGEOFIK_TARGET_CLONES
unsigned int franka_ik_q6(const array<double, 3>& r,
                          const array<double, 9>& ROE, const double q6,
                          array<array<double, 7>, 8>& qsols,
                          const SolverTuning& tuning) {
  const double q1_sing = tuning.q1_sing;
  const double q7_sing = tuning.q7_sing;
  // IK with q6 as free variable
  // INPUT: r = r_EO_O, position of frame E in frame O
  //        ROE, orientation of frame E in frame O (row-first format)
  //        q6, joint angle of joint 6
  //        qsols, array to store 8 solutions
  //        q1_sing, emergency value of q1 in case of singularity at shoulder
  //        joints (type-1 singularity). q7_sing, emergency value of q7 in case
  //        of singularity of S7 intersecting S (type-2 singularity).
  // OUTPUT: number of solutions found.
  // NOTATION:
  // ri = r_iS_O, i = 1,2,3,4,5,6,7
  // si = s_i_O
  array<double, 3> r_ES_O = {r[0], r[1], r[2] - d1};
  array<double, 3> tmp_v = {r_ES_O[1] * ROE[8] - r_ES_O[2] * ROE[5],
                            r_ES_O[2] * ROE[2] - r_ES_O[0] * ROE[8],
                            r_ES_O[0] * ROE[5] - r_ES_O[1] * ROE[2]};
  if (tmp_v[0] * tmp_v[0] + tmp_v[1] * tmp_v[1] + tmp_v[2] * tmp_v[2] <
      SING_TOL * SING_TOL)
    return franka_ik_q7(r, ROE, q7_sing, qsols, tuning);
  const double sg1 = sin(q6);  // sin(pi-q6)
  const double cos_q6 = cos(q6);
  // Wrist nearly parallel to k_E: the closed form below divides by sin(q6),
  // and its cone step subtracts two O(a7/sin(q6)) quantities whose
  // difference is O(sin(q6)), so at |sin(q6)| < SING_TOL it carries only
  // ~5 significant digits. The parallel construction works with O(1)
  // quantities and stays exact across this whole sliver, so it goes first.
  // It can fail to assemble for individual poses, though; those fall
  // through to the closed form so coverage never regresses.
  if (sg1 * sg1 < WRIST_PARALLEL_TOL * WRIST_PARALLEL_TOL) {
    const unsigned int n_parallel =
        franka_ik_q6_parallel(r_ES_O, ROE, cos_q6 >= 0 ? 1 : -1, qsols, tuning);
    if (n_parallel > 0) return n_parallel;
  }

  // NON-PARALLEL CASE:
  array<double, 3> s7 = {ROE[2], ROE[5], ROE[8]};
  const double cg1 = -cos_q6;  // cos(pi-q6)
  array<double, 3> r_O7S_O = {r_ES_O[0] - dE * ROE[2], r_ES_O[1] - dE * ROE[5],
                              r_ES_O[2] - dE * ROE[8]};
  array<double, 3> r_PS_O = {r_O7S_O[0] + (a7 * cg1 / sg1) * s7[0],
                             r_O7S_O[1] + (a7 * cg1 / sg1) * s7[1],
                             r_O7S_O[2] + (a7 * cg1 / sg1) * s7[2]};
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
      nanogeofik_log() << "ERROR: unable to assembly kinematic chain";
      for (int i = 0; i < 8; ++i) {
        fill(qsols[i].begin(), qsols[i].end(), NAN);
      }
      return 0;
    }
  }
  // sin^2(tau) = (b1-(lP-c))*(b1+(lP-c))*(lP+c-b1)*(lP+c+b1) / (2*lP*c)^2
  // is Heron's formula for this triangle.  Every factor is a difference of
  // like-sized lengths, so unlike 1 - tmp*tmp it keeps full relative
  // precision as the triangle flattens (tmp -> +-1) -- exactly the
  // wrist-near-parallel regime where the construction divides by sin(q6).
  const double lP_minus_c = lP - c;
  const double lP_plus_c = lP + c;
  const double sin_tau =
      sqrt(std::max(0.0, (b1 - lP_minus_c) * (b1 + lP_minus_c) *
                             (lP_plus_c - b1) * (lP_plus_c + b1))) /
      (2 * lP * c);
  unsigned int n_gamma_sols = 1;
  if ((d3 + d5 + lC < lP) && (lP < b1 + c)) n_gamma_sols = 2;
  // gamma2 is base+tau for the normal elbow assembly and base-tau for
  // the rare second assembly.  cos(gamma2) and |sin(gamma2)| are composed
  // directly from the triangle ratios instead of evaluating acos, atan,
  // and then cos/sin again; composing both avoids the complement-of-a-
  // complement loss in 1 - cos(gamma2)^2, which is what keeps the fourth
  // joint axis unit-length here.
  const double base_numerator = d5 + lC;
  double cos_gamma2s[2] = {(base_numerator * tmp - a5 * sin_tau) / c,
                           (base_numerator * tmp + a5 * sin_tau) / c};
  double sin_gamma2s[2] = {(sin_tau * base_numerator + tmp * a5) / c,
                           (sin_tau * base_numerator - tmp * a5) / c};
  const double cone_denominator = fabs(sg1) * sqrt(Cx * Cx + Cy * Cy);
  const double u2_x = Cx * sg1;
  const double u2_y = Cy * sg1;
  const double u2 = nanogeofik_atan::atan2(u2_x, u2_y);
  const double sin_u2 = u2_x / cone_denominator;
  const double cos_u2 = u2_y / cone_denominator;

  array<array<double, 3>, 4> s5s;
  array<double, 4> inverse_s4_norms;
  double q7s[4];
  double d, u1;
  unsigned int n_sols = 0;
  for (unsigned int i = 0; i < n_gamma_sols; i++) {
    d = lP * cos_gamma2s[i];
    tmp = (d + Cz * cg1) / cone_denominator;
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
    const double sin_u1 = tmp;
    const double cos_u1 = sqrt(std::max(0.0, 1.0 - sin_u1 * sin_u1));
    // cout << "u1 = " << u1 << endl;
    // cout << "u2 = " << u2 << endl;
    q7s[n_sols] = 5 * PI / 4 - u1 + u2;
    tmp_v = {-sg1 * (cos_u1 * cos_u2 + sin_u1 * sin_u2),
             -sg1 * (sin_u1 * cos_u2 - cos_u1 * sin_u2), cg1};
    column_1s_times_vec(ROE, tmp_v, s5s[n_sols]);
    const double inverse_s4_norm = 1.0 / (lP * fabs(sin_gamma2s[i]));
    inverse_s4_norms[n_sols] = inverse_s4_norm;
    n_sols++;
    q7s[n_sols] = PI / 4 + u1 + u2;
    tmp_v = {sg1 * (cos_u1 * cos_u2 - sin_u1 * sin_u2),
             -sg1 * (sin_u1 * cos_u2 + cos_u1 * sin_u2), cg1};
    column_1s_times_vec(ROE, tmp_v, s5s[n_sols]);
    inverse_s4_norms[n_sols] = inverse_s4_norm;
    n_sols++;
  }

  array<double, 3> s2, s3, s4, s6, r4, r6;
  array<double, 6> sol1;
  array<double, 3> sol2;
  unsigned int assembled_sols = 0;
  for (unsigned int i = 0; i < n_sols; i++) {
    r6 = {r_PS_O[0] - lC * s5s[i][0], r_PS_O[1] - lC * s5s[i][1],
          r_PS_O[2] - lC * s5s[i][2]};
    tmp_v = {r_O7S_O[0] - r6[0], r_O7S_O[1] - r6[1], r_O7S_O[2] - r6[2]};
    Cross_(s7, tmp_v, s6);
    s6 = {s6[0] * inv_a7, s6[1] * inv_a7, s6[2] * inv_a7};
    Cross_(s5s[i], r6, s4);
    s4 = {s4[0] * inverse_s4_norms[i], s4[1] * inverse_s4_norms[i],
          s4[2] * inverse_s4_norms[i]};
    Cross_(s5s[i], s4, tmp_v);
    r4 = {r6[0] - d5 * s5s[i][0] + a5 * tmp_v[0],
          r6[1] - d5 * s5s[i][1] + a5 * tmp_v[1],
          r6[2] - d5 * s5s[i][2] + a5 * tmp_v[2]};
    tmp = Norm(r4);
    // r4 is the shoulder-to-joint-4 vector and must have the fixed upper-arm
    // length. Cone roots that violate this invariant are algebraic artifacts.
    if (fabs(tmp - b1) > SING_TOL) continue;
    rotate_by_sin_cos(s4, sin_beta1, cos_beta1, r4, s3);
    s3 = {s3[0] / tmp, s3[1] / tmp, s3[2] / tmp};
    tmp = s3[1] * s3[1] + s3[0] * s3[0];
    if (tmp > SHOULDER_SING_TOL * SHOULDER_SING_TOL)
      s2 = shoulder_axis_from_s3(s3, tmp);
    else
      s2 = {sin(q1_sing), cos(q1_sing), 0};
    sol1 = q_from_axes_with_q6(s2, s3, s4, s5s[i], s6, q6);
    sol2 = q_from_flipped_shoulder(sol1);
    qsols[2 * assembled_sols] = {sol1[0], sol1[1], sol1[2], sol1[3],
                                 sol1[4], sol1[5], q7s[i]};
    check_limits(qsols[2 * assembled_sols], 7, tuning);
    qsols[2 * assembled_sols + 1] = {sol2[0],
                                     sol2[1],
                                     sol2[2],
                                     qsols[2 * assembled_sols][3],
                                     qsols[2 * assembled_sols][4],
                                     qsols[2 * assembled_sols][5],
                                     qsols[2 * assembled_sols][6]};
    check_limits(qsols[2 * assembled_sols + 1], 3, tuning);
    assembled_sols++;
  }
  for (int i = 2 * assembled_sols; i < 8; ++i) {
    fill(qsols[i].begin(), qsols[i].end(), NAN);
  }
  return 2 * assembled_sols;
}

// FUNCTIONS FOR Q5 LOCKING
//
// The paper (Sec. 2) leaves the solver with q5 as free variable to future
// work because "it is geometrically the most complex", and judges that
// eliminating the wrist-cone angle "leaves a single scalar equation in the
// wrist-circle angle delta = pi/4 - q7 whose algebraic degree rules out a
// radical solution".  The equation does reduce, and in the right variable it
// reduces all the way to a quartic, so the solver here is closed-form.
//
// Frame C is built as in Sec. 3: k_C = -r6/l, i_C = (k_C x s6)/|k_C x s6|,
// j_C = k_C x i_C, with the solved wrist axes written as
//
//   s6 = (0, sy, sz)_C,     s5 = (-sa2 cos(g), -sa2 sin(g), -ca2)_C,
//
// where sa2/ca2 come from the elbow triangle S-O4-O6 and g is the cone
// angle.  The wrist closure s5.s6 = 0 gives sin(g) = -sz*ca2/(sy*sa2), and
// the locked-joint definition q5 = atan2((s4 x s6).s5, s4.s6), together
// with s4 = s5 x r6/(l*|sa2|), gives the exact identities
//
//   sin(q5) = sz / sa2      and      cos(q5) = -sy*cos(g).
//
// With S5 := sin(q5) fixed by the caller, the first identity is a scalar
// equation in delta alone,
//
//   F(delta) = sz(delta) - sa2(delta)*S5 = 0,
//   sz(delta) = -(s6.r_O7S)/l(delta).
//
// The reduction to a polynomial turns on writing that equation in
//
//   u = delta - psi,   w = cos(u),
//   psi = atan2(r_O7S.j_E, r_O7S.i_E),   h = |(r_O7S.i_E, r_O7S.j_E)|,
//
// because both quantities F is built from are then rational in w up to two
// square roots.  With l^2 = A - B*w (A = |r_O7S|^2 + a7^2, B = 2*a7*h),
//
//   sz  = h*sin(u)/l,
//   sa2 = [sin(b2)*(l^2 - K) + s*cos(b2)*sqrt(D)] / (2*b2*l),
//   K   = b1^2 - b2^2,   D = 4*b2^2*l^2 - (l^2 - K)^2,
//
// where s = +-1 selects the elbow assembly.  The single power of l in sa2's
// denominator is exactly the one sz carries, so multiplying F = 0 through by
// 2*b2*l clears it completely:
//
//   2*b2*h*sin(u) = S5*[sin(b2)*(l^2 - K) + s*cos(b2)*sqrt(D)].
//
// What is left is linear in w (l^2 - K), quadratic in w (D), and carries the
// two radicals sin(u) = e*sqrt(1 - w^2) and sqrt(D).  Squaring twice removes
// both and leaves, with T(w) := Q^2*(1 - w^2) + p(w)^2 - M^2*D(w) quadratic
// in w,
//
//   T(w)^2 - 4*Q^2*p(w)^2*(1 - w^2) = 0,
//   Q = 2*b2*h,   p(w) = S5*sin(b2)*(l^2 - K),   M = S5*cos(b2),
//
// a quartic in w whose root set contains every root of F for both elbow
// assemblies and both signs of sin(u) at once.  One quartic therefore
// replaces the whole scan: its real roots in [-1, 1] are enumerated exactly,
// each gives the two candidate angles u = +-acos(w), and evaluating F picks
// out which of the four sign combinations each candidate actually solves.
//
// An assembled arm must reproduce the requested joint value this closely.
constexpr double Q5_LOCK_VERIFY_TOL = 1e-7;
// Roots closer than this on the same branch are the same solution.
constexpr double Q5_ROOT_DUP_TOL = 1e-9;
// Solutions closer than this (per joint, wrapped) are duplicates.
constexpr double Q5_SOLUTION_DUP_TOL = 1e-8;
// Defensive cap on roots; each yields at most two solutions (shoulder flip).
constexpr unsigned int Q5_MAX_ROOTS = 8;
// A candidate whose residual is still this large is a wrong sign combination
// from the double squaring rather than a root: those sit at O(1), so the gate
// only has to separate zero from that.  Everything past it is refined and then
// judged by the assemble-and-verify gate.
constexpr double Q5_ROOT_CANDIDATE_TOL = 1e-2;
// A polished candidate at or below this has reached the residual's own noise
// floor and is a simple root.  Above it, the candidate sits near a critical
// point of F, where Newton stalls and the pair has to be resolved directly.
constexpr double Q5_ROOT_CONVERGED_TOL = 1e-12;
// A stalled candidate this close to zero is a genuine repeated root: F touches
// the level rather than crossing it, and the two solutions have merged.
constexpr double Q5_TANGENCY_ACCEPT_TOL = 1e-9;
// Relative slack for reading a double root off a critical point of the
// quartic.  Deliberately loose: emitting a candidate that turns out not to be
// a root costs one residual evaluation, missing a merged pair of solutions
// costs correctness.
constexpr double Q5_DOUBLE_ROOT_TOL = 1e-7;

// remainder(x, 2*pi) without the libm call.  Every comparison the solver
// makes against the result is against a tolerance many orders of magnitude
// above the rounding this introduces.
static inline double q5_wrap_two_pi(const double x) {
  return x - TWO_PI * std::rint(x * (1.0 / TWO_PI));
}

struct Q5Root {
  double delta;
  double cos_delta;
  double sin_delta;
  unsigned int branch;  // elbow assembly: 0 -> beta2+actmp, 1 -> beta2-actmp.
};

// Shared scalars of a root solve: projections of r_O7S on the wrist-circle
// basis and the requested sin(q5).
struct Q5ScanCtx {
  double ro_iE;
  double ro_jE;
  double ro_R2;
  double sin_q5;
};

// F(delta) for one elbow assembly; false where the chain cannot assemble.
static inline bool q5_residual(const double cos_d, const double sin_d,
                               const double ro_iE, const double ro_jE,
                               const double ro_R2, const double sin_q5,
                               const unsigned int branch, double& f_out) {
  const double ro_i6 = ro_iE * cos_d + ro_jE * sin_d;
  const double l2 = ro_R2 + a7 * a7 - 2.0 * a7 * ro_i6;
  if (!(l2 > 1e-12)) return false;
  const double l = sqrt(l2);
  // The elbow-down assembly only exists inside the flat window, exactly as
  // in franka_ik_q7().  The window coincides with sin(alpha2) > 0; beyond
  // it every assembly has q4 >= 0 and violates the Panda joint limits (see
  // the note in franka_ik_q7()), so enumerating it would only produce rows
  // that check_limits() rejects.
  if (branch == 1 && !(d3 + d5 < l && l < b1 + b2)) return false;
  const double inv_l = 1.0 / l;
  // cos(angle at O6) = (l^2 - (b1^2 - b2^2)) / (2 l b2)
  double tmp = (l2 - (b1 * b1 - b2 * b2)) / (2.0 * b2 * l);
  if (!clamp_trig_roundoff(tmp)) return false;
  const double sin_actmp = sqrt(std::max(0.0, 1.0 - tmp * tmp));
  const double sa2 = branch == 0 ? sin_beta2 * tmp + cos_beta2 * sin_actmp
                                 : sin_beta2 * tmp - cos_beta2 * sin_actmp;
  // sz = s6 . k_C = -(s6 . r6)/l, using s6 . i_6 = 0 and
  // s6 = cos(d)*j_E - sin(d)*i_E.
  const double sz = (ro_iE * sin_d - ro_jE * cos_d) * inv_l;
  f_out = sz - sa2 * sin_q5;
  return true;
}

// Both elbow assemblies at one abscissa.  Everything up to the cone term is
// common to the two, so the pair costs one extra add over a single residual.
static inline void q5_residual_pair(const Q5ScanCtx& ctx, const double cos_d,
                                    const double sin_d, double* f_out,
                                    bool* valid_out, double& l_out) {
  valid_out[0] = false;
  valid_out[1] = false;
  l_out = 0.0;
  const double ro_i6 = ctx.ro_iE * cos_d + ctx.ro_jE * sin_d;
  const double l2 = ctx.ro_R2 + a7 * a7 - 2.0 * a7 * ro_i6;
  if (!(l2 > 1e-12)) return;
  const double l = sqrt(l2);
  l_out = l;
  const double inv_l = 1.0 / l;
  double tmp = (l2 - (b1 * b1 - b2 * b2)) / (2.0 * b2 * l);
  if (!clamp_trig_roundoff(tmp)) return;
  const double sin_actmp = sqrt(std::max(0.0, 1.0 - tmp * tmp));
  const double sz = (ctx.ro_iE * sin_d - ctx.ro_jE * cos_d) * inv_l;
  const double base = sin_beta2 * tmp;
  const double cone = cos_beta2 * sin_actmp;
  f_out[0] = sz - (base + cone) * ctx.sin_q5;
  valid_out[0] = true;
  if (d3 + d5 < l && l < b1 + b2) {
    f_out[1] = sz - (base - cone) * ctx.sin_q5;
    valid_out[1] = true;
  }
}

// F and dF/ddelta together, for the Newton polish.  Differentiating the
// residual through l gives, with hs = h*sin(u) and ro_i6 = h*cos(u),
//
//   dl2 = 2*a7*hs,   dl = a7*hs/l,
//   d(sz) = ro_i6/l - hs*dl/l^2,
//   d(tmp) = (l^2 + K)/(2*b2*l^2) * dl,
//   d(sa2) = [sin(b2) -+ cos(b2)*tmp/sin(actmp)] * d(tmp).
//
// The slope diverges where the elbow triangle goes flat (sin(actmp) -> 0);
// the caller's step guard is what keeps that harmless.
static inline bool q5_residual_slope(const Q5ScanCtx& ctx, const double x,
                                     const unsigned int branch, double& f_out,
                                     double& df_out) {
  double cos_d, sin_d;
  nanogeofik_sincos::sincos(x, sin_d, cos_d);
  const double ro_i6 = ctx.ro_iE * cos_d + ctx.ro_jE * sin_d;
  const double hs = ctx.ro_iE * sin_d - ctx.ro_jE * cos_d;
  const double l2 = ctx.ro_R2 + a7 * a7 - 2.0 * a7 * ro_i6;
  if (!(l2 > 1e-12)) return false;
  const double l = sqrt(l2);
  if (branch == 1 && !(d3 + d5 < l && l < b1 + b2)) return false;
  const double inv_l = 1.0 / l;
  constexpr double kTriK = b1 * b1 - b2 * b2;
  double tmp = (l2 - kTriK) / (2.0 * b2 * l);
  if (!clamp_trig_roundoff(tmp)) return false;
  const double sin_actmp = sqrt(std::max(0.0, 1.0 - tmp * tmp));
  const double signed_cos_beta2 = branch == 0 ? cos_beta2 : -cos_beta2;
  const double sa2 = sin_beta2 * tmp + signed_cos_beta2 * sin_actmp;
  const double sz = hs * inv_l;
  f_out = sz - sa2 * ctx.sin_q5;
  const double dl = a7 * hs * inv_l;
  const double dsz = ro_i6 * inv_l - hs * dl * inv_l * inv_l;
  const double dtmp = (l2 + kTriK) * inv_l * inv_l * dl / (2.0 * b2);
  // sin(actmp) in the denominator is the flat-triangle blow-up; report the
  // slope as unusable there rather than manufacturing an infinity.
  if (!(sin_actmp > 1e-12)) return false;
  const double dsa2 = (sin_beta2 - signed_cos_beta2 * tmp / sin_actmp) * dtmp;
  df_out = dsz - dsa2 * ctx.sin_q5;
  return true;
}

static void q5_push_root(Q5Root* roots, unsigned int& n_roots,
                         const Q5Root& candidate) {
  for (unsigned int k = 0; k < n_roots; ++k)
    if (roots[k].branch == candidate.branch &&
        fabs(q5_wrap_two_pi(candidate.delta - roots[k].delta)) <
            Q5_ROOT_DUP_TOL)
      return;
  if (n_roots < Q5_MAX_ROOTS) roots[n_roots++] = candidate;
}

// Records the abscissa of a root together with its trigonometrics.
static inline void q5_set_root(Q5Root& out, const double delta,
                               const unsigned int branch) {
  out.delta = delta;
  nanogeofik_sincos::sincos(delta, out.sin_delta, out.cos_delta);
  out.branch = branch;
}

// Newton on the exact residual.  The quartic lands within ~1e-8 of a simple
// root, so two or three steps reach the floor of F itself; a step that leaves
// the local basin, stops improving, or hits the flat-triangle slope ends the
// polish and keeps the best point seen.
static bool q5_polish_root(const Q5ScanCtx& ctx, const unsigned int branch,
                           double& delta, double& f_out) {
  double x = delta;
  double f, df;
  if (!q5_residual_slope(ctx, x, branch, f, df)) {
    // No usable slope here (flat triangle, or just outside the assembly
    // window).  Fall back to the plain residual so the caller can still
    // judge the candidate.
    double c, s;
    nanogeofik_sincos::sincos(x, s, c);
    if (!q5_residual(c, s, ctx.ro_iE, ctx.ro_jE, ctx.ro_R2, ctx.sin_q5, branch,
                     f))
      return false;
    f_out = f;
    return true;
  }
  for (int it = 0; it < 8; ++it) {
    if (!(fabs(df) > 0.0)) break;
    double step = f / df;
    // The candidate is already close; a large step means the slope is not
    // describing this root.
    if (!(fabs(step) < 1e-2)) break;
    // A root can sit hard against an assembly window edge -- the elbow-down
    // window is only 19 mm wide in l -- so a full Newton step may land where
    // the chain no longer assembles.  Shortening it walks up to the edge
    // instead of abandoning the root there.
    double xn = x - step;
    double fn, dfn;
    bool ok = q5_residual_slope(ctx, xn, branch, fn, dfn);
    for (int back = 0; !ok && back < 8; ++back) {
      step *= 0.5;
      xn = x - step;
      if (xn == x) break;
      ok = q5_residual_slope(ctx, xn, branch, fn, dfn);
    }
    if (!ok) break;
    if (!(fabs(fn) < fabs(f))) {
      // Converged to the residual's floor.
      if (fabs(fn) <= fabs(f)) {
        x = xn;
        f = fn;
      }
      break;
    }
    x = xn;
    f = fn;
    df = dfn;
  }
  delta = x;
  f_out = f;
  return true;
}

// F at one abscissa.
static inline bool q5_residual_at(const Q5ScanCtx& ctx, const double x,
                                  const unsigned int branch, double& f_out) {
  double c, s;
  nanogeofik_sincos::sincos(x, s, c);
  return q5_residual(c, s, ctx.ro_iE, ctx.ro_jE, ctx.ro_R2, ctx.sin_q5, branch,
                     f_out);
}

// Bisects a bracketed sign change of F down to floating-point resolution.
static bool q5_bisect_root(double da, double fa, double db,
                           const Q5ScanCtx& ctx, const unsigned int branch,
                           Q5Root& out) {
  for (int it = 0; it < 100; ++it) {
    const double mid = 0.5 * (da + db);
    if (mid == da || mid == db) break;
    double fm;
    // Valid endpoints with opposite signs guarantee an assembling interior,
    // so a failing evaluation means the bracket straddled an assembly
    // boundary and the sign change was spurious.
    if (!q5_residual_at(ctx, mid, branch, fm)) return false;
    if ((fm <= 0.0) == (fa <= 0.0)) {
      da = mid;
      fa = fm;
    } else {
      db = mid;
    }
  }
  q5_set_root(out, 0.5 * (da + db), branch);
  return true;
}

// Resolves a candidate that Newton could not drive to the residual floor.
//
// Such a candidate sits next to a critical point of F, which means one of
// three things: F dips through the level twice in a span too narrow for the
// quartic to separate (the quartic's own roots are ill conditioned around a
// double root, moving like the square root of a coefficient perturbation),
// F touches the level exactly and the two solutions have merged, or F misses
// the level and there is no root at all.  The residual itself separates the
// three cheaply: probe outwards on a widening ladder, and a sign change found
// on either side brackets a genuine root that bisection then pins to full
// precision.  Returns true if any root was pushed.
static bool q5_resolve_stalled(const Q5ScanCtx& ctx, const unsigned int branch,
                               const double x0, const double f0, Q5Root* roots,
                               unsigned int& n_roots) {
  static constexpr double kLadder[] = {1e-10, 1e-9, 1e-8, 1e-7, 1e-6,
                                       1e-5,  1e-4, 1e-3, 1e-2};
  bool found = false;
  for (int side = 0; side < 2; ++side) {
    const double dir = side == 0 ? 1.0 : -1.0;
    double xa = x0, fa = f0;
    for (const double step : kLadder) {
      const double xb = x0 + dir * step;
      double fb;
      if (!q5_residual_at(ctx, xb, branch, fb)) break;
      if ((fb < 0.0) != (fa < 0.0)) {
        Q5Root bracketed;
        if (q5_bisect_root(xa, fa, xb, ctx, branch, bracketed)) {
          q5_push_root(roots, n_roots, bracketed);
          found = true;
        }
        break;
      }
      xa = xb;
      fa = fb;
    }
  }
  return found;
}

// The radii at which an assembly window opens or closes.  The elbow triangle
// goes flat at |b1 - b2| and at b1 + b2, and the elbow-down assembly is
// additionally confined to l > d3 + d5.
constexpr double Q5_WINDOW_EDGE_MARGIN = 1e-4;

static inline bool q5_near_window_edge(const double l) {
  return fabs(l - (b1 + b2)) < Q5_WINDOW_EDGE_MARGIN ||
         fabs(l - (d3 + d5)) < Q5_WINDOW_EDGE_MARGIN ||
         fabs(l - fabs(b1 - b2)) < Q5_WINDOW_EDGE_MARGIN;
}

// Steps a candidate that fell just outside an assembly window back inside it.
//
// The quartic's roots are ill conditioned exactly where the two elbow
// assemblies merge, because that is where the radical sqrt(D) vanishes and the
// squared equation picks up a double root: w is then recoverable only to about
// the square root of machine precision, a few times 1e-8 in delta.  That is
// normally far below anything that matters, but the elbow-down assembly exists
// only for l in (d3 + d5, b1 + b2) -- a 19 mm window -- and a root pressed
// against its flat-triangle edge can have assembling ground just 1e-8 wide on
// one side of it.  Landing on the wrong side of that edge must not discard the
// root, so walk outwards until the chain assembles and let the polish finish.
static bool q5_step_into_window(const Q5ScanCtx& ctx, const unsigned int branch,
                                double& delta, double& f_out) {
  static constexpr double kLadder[] = {1e-10, 1e-9, 1e-8, 1e-7,
                                       1e-6,  1e-5, 1e-4};
  for (const double step : kLadder) {
    for (int side = 0; side < 2; ++side) {
      const double x = delta + (side == 0 ? step : -step);
      double f;
      if (q5_residual_at(ctx, x, branch, f)) {
        delta = x;
        f_out = f;
        return true;
      }
    }
  }
  return false;
}

// Real roots of a monic cubic t^3 + a2*t^2 + a1*t + a0, ascending.
static int q5_cubic_real_roots(const double a2, const double a1,
                               const double a0, double* out) {
  // Depress with t = y - a2/3.
  const double shift = a2 / 3.0;
  const double p = a1 - a2 * shift;
  const double q = a0 - shift * (a1 - 2.0 * a2 * a2 / 9.0);
  const double half_q = 0.5 * q;
  const double third_p = p / 3.0;
  const double disc = half_q * half_q + third_p * third_p * third_p;
  int n = 0;
  if (disc > 0.0) {
    const double root_disc = sqrt(disc);
    out[n++] =
        std::cbrt(-half_q + root_disc) + std::cbrt(-half_q - root_disc) - shift;
  } else {
    // Three real roots (or a repeated one).  The trigonometric form stays
    // conditioned where Cardano's two cube roots would cancel.
    const double radius = sqrt(std::max(0.0, -third_p));
    if (!(radius > 0.0)) {
      out[n++] = -shift;
    } else {
      double arg = -half_q / (radius * radius * radius);
      if (arg > 1.0) arg = 1.0;
      if (arg < -1.0) arg = -1.0;
      const double phi = acos(arg) / 3.0;
      const double scale = 2.0 * radius;
      // The three roots sit 120 degrees apart, so one sincos and the rotation
      // identities cover all of them:
      //   cos(phi -+ 2pi/3) = -cos(phi)/2 +- sin(phi)*sqrt(3)/2.
      double sin_phi, cos_phi;
      nanogeofik_sincos::sincos(phi, sin_phi, cos_phi);
      constexpr double kHalfRoot3 = 8.66025403784438646764e-01;
      const double half_cos = -0.5 * cos_phi;
      const double wing = kHalfRoot3 * sin_phi;
      out[n++] = scale * cos_phi - shift;
      out[n++] = scale * (half_cos + wing) - shift;
      out[n++] = scale * (half_cos - wing) - shift;
    }
  }
  std::sort(out, out + n);
  return n;
}

// Real roots of c4*w^4 + c3*w^3 + c2*w^2 + c1*w + c0 inside [-1, 1].
//
// Isolation rather than a radical formula, because completeness is what
// matters here and Ferrari's nested radicals lose it: the roots of the
// derivative cut [-1, 1] into intervals on which the quartic is monotone, so
// a sign change across one brackets exactly one root, and a critical point
// where the quartic is already near zero is a double root.  Every real root
// in range is therefore accounted for.  Each is only refined far enough for
// Newton on the true residual to finish the job.
static int q5_quartic_roots(const double c4, const double c3, const double c2,
                            const double c1, const double c0, double* out) {
  const double dc3 = 4.0 * c4, dc2 = 3.0 * c3, dc1 = 2.0 * c2;
  double crit[3];
  int n_crit = 0;
  if (fabs(dc3) > 0.0)
    n_crit = q5_cubic_real_roots(dc2 / dc3, dc1 / dc3, c1 / dc3, crit);
  double bp[5];
  int n_bp = 0;
  bp[n_bp++] = -1.0;
  for (int i = 0; i < n_crit; ++i)
    if (crit[i] > -1.0 && crit[i] < 1.0) bp[n_bp++] = crit[i];
  bp[n_bp++] = 1.0;
  for (int i = 1; i < n_bp; ++i) {
    const double key = bp[i];
    int j = i - 1;
    while (j >= 0 && bp[j] > key) {
      bp[j + 1] = bp[j];
      --j;
    }
    bp[j + 1] = key;
  }
  double scale = fabs(c0);
  scale = std::max(scale, fabs(c1));
  scale = std::max(scale, fabs(c2));
  scale = std::max(scale, fabs(c3));
  scale = std::max(scale, fabs(c4));
  if (!(scale > 0.0)) scale = 1.0;
  int n = 0;
  auto poly = [&](const double w) {
    return (((c4 * w + c3) * w + c2) * w + c1) * w + c0;
  };
  auto push = [&](double w) {
    if (w < -1.0) w = -1.0;
    if (w > 1.0) w = 1.0;
    for (int k = 0; k < n; ++k)
      if (fabs(out[k] - w) < 1e-12) return;
    if (n < 8) out[n++] = w;
  };
  // One evaluation per breakpoint rather than two per interval.
  double f_bp[5];
  for (int k = 0; k < n_bp; ++k) f_bp[k] = poly(bp[k]);
  for (int k = 0; k + 1 < n_bp; ++k) {
    double lo = bp[k], hi = bp[k + 1];
    double f_lo = f_bp[k], f_hi = f_bp[k + 1];
    if (f_lo == 0.0) push(lo);
    if (f_hi == 0.0) push(hi);
    if ((f_lo < 0.0) == (f_hi < 0.0)) continue;
    // Monotone here, so exactly one root.  Newton converges quadratically on
    // a monotone span; keeping the bracket and falling back to its midpoint
    // whenever a step would leave it makes that safe without paying for a
    // full bisection.
    double w = 0.5 * (lo + hi);
    for (int it = 0; it < 24; ++it) {
      const double f_w = poly(w);
      if (f_w == 0.0) break;
      const double slope = ((4.0 * c4 * w + 3.0 * c3) * w + 2.0 * c2) * w + c1;
      // Halley: cubic convergence for one extra second-derivative evaluation,
      // which pays for itself on the first steps away from the midpoint.
      const double curv = (12.0 * c4 * w + 6.0 * c3) * w + 2.0 * c2;
      const double denom = 2.0 * slope * slope - f_w * curv;
      const double next_newton =
          fabs(denom) > 0.0
              ? w - 2.0 * f_w * slope / denom
              : (fabs(slope) > 0.0 ? w - f_w / slope : 0.5 * (lo + hi));
      // Convergence has to be judged before the bracket takes w as an
      // endpoint: once the Newton step rounds away to nothing the root is
      // resolved, but the tightened bracket would then reject that step as
      // "outside" and throw the iterate away for a bisection restart.
      if (next_newton == w) break;
      if ((f_w < 0.0) == (f_lo < 0.0)) {
        lo = w;
        f_lo = f_w;
      } else {
        hi = w;
      }
      // Newton wherever it stays inside the bracket, bisection otherwise.
      double next = next_newton;
      if (!(next > lo && next < hi)) next = 0.5 * (lo + hi);
      if (next == w) break;
      // The root only has to be located well enough that the residual gate
      // downstream either accepts it outright or brackets it; resolving the
      // quartic to the last bit buys nothing.
      const bool tight = fabs(next - w) <= 1e-12 * (1.0 + fabs(w));
      w = next;
      if (tight) break;
    }
    push(w);
  }
  // Double roots never show up as a sign change; they sit at a critical
  // point where the quartic is already near zero.
  for (int i = 0; i < n_crit; ++i) {
    if (!(crit[i] >= -1.0 && crit[i] <= 1.0)) continue;
    if (fabs(poly(crit[i])) <= Q5_DOUBLE_ROOT_TOL * scale) push(crit[i]);
  }
  return n;
}

// Collects the roots of F(delta) = sz - sa2*sin(q5): the locking equation of
// the q5 solver.  See the derivation above -- one quartic in w = cos(delta -
// psi) carries every root of both elbow assemblies, so this enumerates its
// real roots in [-1, 1], turns each into the two candidate angles it stands
// for, and keeps the ones the residual confirms.  Returns the number of roots
// found (at most Q5_MAX_ROOTS).
static unsigned int find_q5_roots(const array<double, 3>& i_E_O,
                                  const array<double, 3>& j_E_O,
                                  const array<double, 3>& r_O7S_O,
                                  const double sin_q5, Q5Root* roots,
                                  const SolverTuning& tuning) {
  const Q5ScanCtx ctx = {Dot(r_O7S_O, i_E_O), Dot(r_O7S_O, j_E_O),
                         Dot(r_O7S_O, r_O7S_O), sin_q5};
  // h is |r_O7S x k_E|, which the type-2 gate in the callers has already put
  // above SING_TOL, so the wrist circle is never degenerate here.
  const double h = sqrt(ctx.ro_iE * ctx.ro_iE + ctx.ro_jE * ctx.ro_jE);
  if (!(h > 0.0)) return 0;
  // The wrist-circle basis rotated by psi, as a direction rather than an
  // angle: cos(psi) = (r_O7S.i_E)/h, sin(psi) = (r_O7S.j_E)/h.  Carrying it
  // this way lets a root in w become (cos, sin) of delta by one rotation,
  // with no inverse cosine and no sincos per candidate.
  const double inv_h = 1.0 / h;
  const double cos_psi = ctx.ro_iE * inv_h;
  const double sin_psi = ctx.ro_jE * inv_h;

  constexpr double kTriK = b1 * b1 - b2 * b2;
  const double A = ctx.ro_R2 + a7 * a7;
  const double B = 2.0 * a7 * h;
  const double G = A - kTriK;
  const double Q = 2.0 * b2 * h;
  const double M = sin_q5 * cos_beta2;
  const double P0 = sin_q5 * sin_beta2 * G;
  const double P1 = -sin_q5 * sin_beta2 * B;
  // D(w) = 4*b2^2*l^2 - (l^2 - K)^2 with l^2 = A - B*w.
  const double D0 = 4.0 * b2 * b2 * A - G * G;
  const double D1 = 2.0 * B * (G - 2.0 * b2 * b2);
  const double D2 = -B * B;
  const double Q2 = Q * Q;
  const double M2 = M * M;
  // T(w) = Q^2*(1 - w^2) + p(w)^2 - M^2*D(w).
  const double T0 = Q2 + P0 * P0 - M2 * D0;
  const double T1 = 2.0 * P0 * P1 - M2 * D1;
  const double T2 = -Q2 + P1 * P1 - M2 * D2;
  // T(w)^2 - 4*Q^2*p(w)^2*(1 - w^2).
  const double c4 = T2 * T2 + 4.0 * Q2 * P1 * P1;
  const double c3 = 2.0 * T1 * T2 + 8.0 * Q2 * P0 * P1;
  const double c2 = T1 * T1 + 2.0 * T0 * T2 - 4.0 * Q2 * (P1 * P1 - P0 * P0);
  const double c1 = 2.0 * T0 * T1 - 8.0 * Q2 * P0 * P1;
  const double c0 = T0 * T0 - 4.0 * Q2 * P0 * P0;

  double ws[8];
  const int n_w = q5_quartic_roots(c4, c3, c2, c1, c0, ws);

  const JointLimits& limits = resolve_limits(tuning);
  // delta = pi/4 - q7 descends from pi/4 - lower to pi/4 - upper; work on
  // the ascending interval between those bounds.
  const double d_lo = PI / 4.0 - limits.upper[6];
  const double d_hi = PI / 4.0 - limits.lower[6];
  unsigned int n_roots = 0;
  for (int i = 0; i < n_w; ++i) {
    double w = ws[i];
    if (w > 1.0) w = 1.0;
    if (w < -1.0) w = -1.0;
    const double abs_sin_u = sqrt(std::max(0.0, 1.0 - w * w));
    for (int sign = 0; sign < 2; ++sign) {
      // The quartic lost which sign of sin(u) the root belongs to; both
      // candidates are offered and the residual decides.
      const double sin_u = sign == 0 ? abs_sin_u : -abs_sin_u;
      const double cos_d = cos_psi * w - sin_psi * sin_u;
      const double sin_d = sin_psi * w + cos_psi * sin_u;
      // Bring the candidate into the sweep window (its width is below 2*pi,
      // so the representative is unique).
      double delta = nanogeofik_atan::atan2(sin_d, cos_d);
      delta -= TWO_PI * std::floor((delta - d_lo) / TWO_PI);
      if (!(delta > d_lo && delta < d_hi)) continue;
      double f_pair[2];
      bool valid_pair[2];
      double l_at_delta;
      q5_residual_pair(ctx, cos_d, sin_d, f_pair, valid_pair, l_at_delta);
      const bool at_edge = q5_near_window_edge(l_at_delta);
      for (unsigned int branch = 0; branch < 2; ++branch) {
        double start = delta;
        double start_cos = cos_d, start_sin = sin_d;
        double f_start = valid_pair[branch] ? f_pair[branch] : 0.0;
        if (!valid_pair[branch]) {
          // Only worth rescuing where an assembly window edge is what made
          // the candidate unusable; anywhere else it is simply not a root.
          if (!at_edge) continue;
          if (!q5_step_into_window(ctx, branch, start, f_start)) continue;
          nanogeofik_sincos::sincos(start, start_sin, start_cos);
        }
        // A wrong sign combination leaves |F| at O(1); only a candidate that
        // is already near a root is worth refining.
        if (!(fabs(f_start) < Q5_ROOT_CANDIDATE_TOL)) continue;
        if (fabs(f_start) <= Q5_ROOT_CONVERGED_TOL) {
          // The common case by a wide margin: the quartic lands within a few
          // ulps of the root, so the residual is already at its floor and
          // there is nothing for Newton to improve.  The rotation that built
          // this abscissa also produced its trigonometrics, so the root
          // carries the very pair the residual was evaluated from.
          Q5Root candidate = {start, start_cos, start_sin, branch};
          q5_push_root(roots, n_roots, candidate);
          continue;
        }
        double refined = start;
        double f_refined;
        if (!q5_polish_root(ctx, branch, refined, f_refined)) continue;
        if (fabs(f_refined) <= Q5_ROOT_CONVERGED_TOL) {
          // Simple root, pinned to the residual's floor.
          Q5Root candidate;
          q5_set_root(candidate, refined, branch);
          q5_push_root(roots, n_roots, candidate);
          continue;
        }
        // Newton could not finish: either a critical point of F is in the
        // way, or an assembly window edge blocked the step.  Either way the
        // residual itself can still bracket the root.
        if (q5_resolve_stalled(ctx, branch, refined, f_refined, roots, n_roots))
          continue;
        // No crossing either side.  Only an exact touch of the level is a
        // solution; anything else is a minimum of |F| that never reaches it.
        if (fabs(f_refined) <= Q5_TANGENCY_ACCEPT_TOL) {
          Q5Root candidate;
          q5_set_root(candidate, refined, branch);
          q5_push_root(roots, n_roots, candidate);
        }
      }
    }
  }
  return n_roots;
}

// Assembles the arm geometry at a root: wrist axes from delta, frame C, the
// cone angles from the locked identities, then the standard shoulder-side
// chain.  Outputs the screw axes needed by both entry points.
static bool assemble_q5_arm(const Q5Root& root, const array<double, 3>& i_E_O,
                            const array<double, 3>& j_E_O,
                            const array<double, 3>& r_O7S_O,
                            const double cos_q5, const double q1_sing,
                            array<double, 3>& s2, array<double, 3>& s3,
                            array<double, 3>& s4, array<double, 3>& s5,
                            array<double, 3>& s6, array<double, 3>& r4,
                            array<double, 3>& r6) {
  const double cd = root.cos_delta;
  const double sd = root.sin_delta;
  const array<double, 3> i_6_O = {cd * i_E_O[0] + sd * j_E_O[0],
                                  cd * i_E_O[1] + sd * j_E_O[1],
                                  cd * i_E_O[2] + sd * j_E_O[2]};
  s6 = {cd * j_E_O[0] - sd * i_E_O[0], cd * j_E_O[1] - sd * i_E_O[1],
        cd * j_E_O[2] - sd * i_E_O[2]};
  r6 = {r_O7S_O[0] - a7 * i_6_O[0], r_O7S_O[1] - a7 * i_6_O[1],
        r_O7S_O[2] - a7 * i_6_O[2]};
  const double l = Norm(r6);
  if (!(l > 1e-9)) return false;
  if (root.branch == 1 && !(d3 + d5 < l && l < b1 + b2)) return false;
  double tmp = (b1 * b1 - l * l - b2 * b2) / (-2.0 * l * b2);
  if (!clamp_trig_roundoff(tmp)) return false;
  const double cos_actmp = tmp;
  const double sin_actmp = sqrt(std::max(0.0, 1.0 - cos_actmp * cos_actmp));
  double sa2, ca2;
  if (root.branch == 0) {
    sa2 = sin_beta2 * cos_actmp + cos_beta2 * sin_actmp;
    ca2 = cos_beta2 * cos_actmp - sin_beta2 * sin_actmp;
  } else {
    sa2 = sin_beta2 * cos_actmp - cos_beta2 * sin_actmp;
    ca2 = cos_beta2 * cos_actmp + sin_beta2 * sin_actmp;
  }
  const array<double, 3> k_C_O = {-r6[0] / l, -r6[1] / l, -r6[2] / l};
  array<double, 3> i_C_O = Cross(k_C_O, s6);
  tmp = Norm(i_C_O);
  // Axis 6 nearly parallel to the S-O6 line degenerates the cone frame.
  if (!(tmp > 1e-12)) return false;
  i_C_O = {i_C_O[0] / tmp, i_C_O[1] / tmp, i_C_O[2] / tmp};
  const array<double, 3> j_C_O = Cross(k_C_O, i_C_O);
  const double sy = Dot(s6, j_C_O);
  const double sz = Dot(s6, k_C_O);
  tmp = -cos_q5 / sy;
  if (!clamp_trig_roundoff(tmp)) return false;
  const double cos_gamma = tmp;
  tmp = -sz * ca2 / (sy * sa2);
  if (!clamp_trig_roundoff(tmp)) return false;
  const double sin_gamma = tmp;
  const double v[3] = {-sa2 * cos_gamma, -sa2 * sin_gamma, -ca2};
  s5 = {i_C_O[0] * v[0] + j_C_O[0] * v[1] + k_C_O[0] * v[2],
        i_C_O[1] * v[0] + j_C_O[1] * v[1] + k_C_O[1] * v[2],
        i_C_O[2] * v[0] + j_C_O[2] * v[1] + k_C_O[2] * v[2]};
  Cross_(s5, r6, s4);
  // The normalization must carry the sign of sa2: the common perpendicular
  // from axis 5 to axis 4 flips with the elbow-down assembly, and only the
  // signed scaling closes the chain (|r4| = b1).  For the elbow-up
  // assembly sa2 is always positive, so this matches franka_ik_q7() there.
  const double inverse_s4_norm = 1.0 / (l * sa2);
  s4 = {s4[0] * inverse_s4_norm, s4[1] * inverse_s4_norm,
        s4[2] * inverse_s4_norm};
  array<double, 3> tmp_v;
  Cross_(s5, s4, tmp_v);
  r4 = {r6[0] - d5 * s5[0] + a5 * tmp_v[0], r6[1] - d5 * s5[1] + a5 * tmp_v[1],
        r6[2] - d5 * s5[2] + a5 * tmp_v[2]};
  rotate_by_beta1_scaled(s4, r4, s3);
  tmp = s3[1] * s3[1] + s3[0] * s3[0];
  if (tmp > SHOULDER_SING_TOL * SHOULDER_SING_TOL)
    s2 = shoulder_axis_from_s3(s3, tmp);
  else
    s2 = {sin(q1_sing), cos(q1_sing), 0};
  return true;
}

NANOGEOFIK_TARGET_CLONES
unsigned int franka_ik_q5(const array<double, 3>& r,
                          const array<double, 9>& ROE, const double q5,
                          array<array<double, 7>, 8>& qsols,
                          const SolverTuning& tuning) {
  const double q1_sing = tuning.q1_sing;
  const double q7_sing = tuning.q7_sing;
  // Closed-form analytical IK with q5 as free variable via quartic reduction.
  // See "FUNCTIONS FOR Q5 LOCKING" above for the geometry. INPUT/OUTPUT
  // conventions as franka_ik_q6().
  const array<double, 3> k_E_O = {ROE[2], ROE[5], ROE[8]};
  const array<double, 3> r_ES_O = {r[0], r[1], r[2] - d1};
  // Type-2 singularity: when axis 7 passes through S, the self-motion spins
  // the arm about S7 without changing q4, q5 or q6, so q5 stops being a
  // usable free variable. Hand the problem to the q7 solver with the
  // user's emergency value, like franka_ik_q6().
  array<double, 3> tmp_v = {r_ES_O[1] * k_E_O[2] - r_ES_O[2] * k_E_O[1],
                            r_ES_O[2] * k_E_O[0] - r_ES_O[0] * k_E_O[2],
                            r_ES_O[0] * k_E_O[1] - r_ES_O[1] * k_E_O[0]};
  if (tmp_v[0] * tmp_v[0] + tmp_v[1] * tmp_v[1] + tmp_v[2] * tmp_v[2] <
      SING_TOL * SING_TOL)
    return franka_ik_q7(r, ROE, q7_sing, qsols, tuning);
  const array<double, 3> i_E_O = {ROE[0], ROE[3], ROE[6]};
  const array<double, 3> j_E_O = {ROE[1], ROE[4], ROE[7]};
  const array<double, 3> r_O7S_O = {r_ES_O[0] - dE * k_E_O[0],
                                    r_ES_O[1] - dE * k_E_O[1],
                                    r_ES_O[2] - dE * k_E_O[2]};
  Q5Root roots[Q5_MAX_ROOTS];
  double sin_q5, cos_q5;
  nanogeofik_sincos::sincos(q5, sin_q5, cos_q5);
  const unsigned int n_roots =
      find_q5_roots(i_E_O, j_E_O, r_O7S_O, sin_q5, roots, tuning);
  array<double, 3> s2, s3, s4, s5, s6, r4, r6;
  array<array<double, 7>, 8> accepted;
  unsigned int n_accepted = 0;
  // Each root fills two rows of an eight-row buffer, so stop at four.  The
  // quartic can hand back up to four roots per elbow assembly, which is more
  // than the eight-solution output can carry.
  for (unsigned int i = 0; i < n_roots && n_accepted + 2 <= 8; ++i) {
    if (!assemble_q5_arm(roots[i], i_E_O, j_E_O, r_O7S_O, cos_q5, q1_sing, s2,
                         s3, s4, s5, s6, r4, r6))
      continue;
    // Guard against spurious scan roots: the assembled arm must actually
    // carry the requested joint value.
    if (fabs(q5_wrap_two_pi(signed_angle(s4, s6, s5) - q5)) >
        Q5_LOCK_VERIFY_TOL)
      continue;
    const array<double, 6> sol1 =
        q_from_axes_with_q5(s2, s3, s4, s5, s6, k_E_O, q5);
    const array<double, 3> sol2 = q_from_flipped_shoulder(sol1);
    const double q7 = PI / 4.0 - roots[i].delta;
    accepted[n_accepted] = {sol1[0], sol1[1], sol1[2], sol1[3],
                            sol1[4], sol1[5], q7};
    accepted[n_accepted + 1] = {sol2[0],
                                sol2[1],
                                sol2[2],
                                accepted[n_accepted][3],
                                accepted[n_accepted][4],
                                accepted[n_accepted][5],
                                accepted[n_accepted][6]};
    // Duplicate arms can only arise where the two elbow assemblies meet at
    // an (almost) flat triangle; drop them instead of reporting repeats.
    bool duplicate = false;
    for (unsigned int k = 0; k < n_accepted && !duplicate; k += 2) {
      bool match = true;
      for (int j = 0; j < 7 && match; ++j) {
        if (fabs(q5_wrap_two_pi(accepted[n_accepted][j] - accepted[k][j])) >
            Q5_SOLUTION_DUP_TOL)
          match = false;
        if (fabs(q5_wrap_two_pi(accepted[n_accepted + 1][j] -
                                accepted[k + 1][j])) > Q5_SOLUTION_DUP_TOL)
          match = false;
      }
      duplicate = match;
    }
    if (duplicate) continue;
    n_accepted += 2;
  }
  for (unsigned int k = 0; k < n_accepted; ++k) {
    qsols[k] = accepted[k];
    check_limits(qsols[k], 7, tuning);
  }
  for (unsigned int k = n_accepted; k < 8; ++k)
    fill(qsols[k].begin(), qsols[k].end(), NAN);
  return n_accepted;
}

// FUNCTIONS FOR SWIVEL ANGLE

// The q7 sweep is rewritten around the observation
// that every quantity it needs from a sample is a scalar, and that each of
// those scalars is an affine function of (cos(delta), sin(delta)) with
// delta = pi/4 - q7.  The upstream construction evaluated the full 3-D build
// per sample -- i_6 and s_6, r_6, a normalized frame-C triad, two cone
// branches assembled as vectors, then cross and dot products per branch --
// which cost nine divisions, four square roots and a dozen vector temporaries
// for two numbers.
//
// The reduction uses:
//
//   i_6 = c*i_E + s*j_E,  s_6 = c*j_E - s*i_E,  r_6 = r_O7S - a7*i_6
//   l^2 = |r_O7S|^2 + a7^2 - 2*a7*(r_O7S . i_6)        (i_6 . i_6 = 1)
//   r_6 . s_6 = r_O7S . s_6                            (i_6 . s_6 = 0)
//   s_6 x r_6 = c*(j_E x r_O7S) - s*(i_E x r_O7S) + a7*k_E
//   |s_5 x r_6| = l*|sin(alpha2)|,  s_5 . r_6 = l*cos(alpha2)
//   r_4 = P*r_6 + Q*s_5   with P, Q independent of the cone branch
//
// and, for the swivel error itself, that the two scaled trig terms the error
// gate needs are linear in r_4:
//
//   n_1 . (r_O7S x r_4)             = (n_1 x r_O7S) . r_4
//   (r_O7S x r_4) . (u_O7S x n_1)   = ((u_O7S x n_1) x r_O7S) . r_4
//
// so the two constant vectors c1 and c2 below replace forming the plane normal
// n_2 at all.  The branch orientation test likewise collapses via
// Binet-Cauchy: (r_O7S x r_4) . (s_5 x r_6) =
// (r_O7S . s_5)(r_4 . r_6) - (r_O7S . r_6)(r_4 . s_5), whose last three
// factors do not depend on the branch.
struct SwivelGeometry {
  array<double, 3> r_O7S_O;
  array<double, 3> c1;  // n1_O x r_O7S_O
  array<double, 3> c2;  // (u_O7S_O x n1_O) x r_O7S_O
  array<double, 3> A;   // j_E_O x r_O7S_O
  array<double, 3> B;   // i_E_O x r_O7S_O
  array<double, 3> a7_k_E;
  double R2;            // |r_O7S_O|^2
  double R2_plus_a7sq;  // |r_O7S_O|^2 + a7^2
  double a7_rO_kE;      // r_O7S_O . (s6 x r6), which is a7 * (r_O7S_O . k_E)
  double rO_iE, rO_jE;
  double c1_iE, c1_jE, c1_rO;
  double c2_iE, c2_jE, c2_rO;
  double theta_sin, theta_cos;
};

// Stereographic reference normal (Elias & Wen, 2024).
// Places the single algorithmic singularity ray at -z (pointing straight down
// into the robot base/table, 100% outside the reachable workspace). With e_t =
// [0, 0, 1] and e_r = [0, 1, 0]:
//   n1(u) = e_r - (u . e_r)/(1 + u_z) * (u + e_t)
static inline array<double, 3> stereographic_n1(const array<double, 3>& u) {
  const double denom = 1.0 + u[2];
  if (denom < 1e-6) {
    return {1.0, 0.0, 0.0};
  }
  const double factor = u[1] / denom;
  return {-factor * u[0], 1.0 - factor * u[1], -factor * (u[2] + 1.0)};
}

static SwivelGeometry build_swivel_geometry(const array<double, 3>& i_E_O,
                                            const array<double, 3>& j_E_O,
                                            const array<double, 3>& n1_O,
                                            const array<double, 3>& u_O7S_O,
                                            const array<double, 3>& r_O7S_O,
                                            const double theta) {
  SwivelGeometry g;
  g.r_O7S_O = r_O7S_O;
  g.c1 = Cross(n1_O, r_O7S_O);
  g.c2 = Cross(Cross(u_O7S_O, n1_O), r_O7S_O);
  g.A = Cross(j_E_O, r_O7S_O);
  g.B = Cross(i_E_O, r_O7S_O);
  const array<double, 3> k_E_O = Cross(i_E_O, j_E_O);
  g.a7_k_E = {a7 * k_E_O[0], a7 * k_E_O[1], a7 * k_E_O[2]};
  g.R2 = Dot(r_O7S_O, r_O7S_O);
  g.R2_plus_a7sq = g.R2 + a7 * a7;
  g.a7_rO_kE = a7 * Dot(r_O7S_O, k_E_O);
  g.rO_iE = Dot(r_O7S_O, i_E_O);
  g.rO_jE = Dot(r_O7S_O, j_E_O);
  g.c1_iE = Dot(g.c1, i_E_O);
  g.c1_jE = Dot(g.c1, j_E_O);
  g.c1_rO = Dot(g.c1, r_O7S_O);
  g.c2_iE = Dot(g.c2, i_E_O);
  g.c2_jE = Dot(g.c2, j_E_O);
  g.c2_rO = Dot(g.c2, r_O7S_O);
  g.theta_sin = sin(theta);
  g.theta_cos = cos(theta);
  return g;
}

// dot(n1, n2) and dot(cross(n1, n2), u_O7S) are a common positive scale times
// cos(phi) and sin(phi) for the measured swivel phi.  Rotating that pair by
// the requested theta tests |theta - phi| against the acceptance band without
// normalising or evaluating atan2; the exact signed angle is only needed for
// the small fraction of accepted samples the interpolation step consumes.
static inline double swivel_error_from_scaled(const double theta_sin,
                                              const double theta_cos,
                                              const double phi_cos_scaled,
                                              const double phi_sin_scaled) {
  const double error_sin_scaled =
      theta_sin * phi_cos_scaled - theta_cos * phi_sin_scaled;
  const double error_cos_scaled =
      theta_cos * phi_cos_scaled + theta_sin * phi_sin_scaled;
  if (error_cos_scaled <= 0.0 ||
      fabs(error_sin_scaled) >= TAN_ERR_THRESH * error_cos_scaled)
    return copysign(1e15, error_sin_scaled);
  return nanogeofik_swivel_atan2(error_sin_scaled, error_cos_scaled);
}

struct SwivelSample {
  array<double, 2> signed_errors;
  array<double, 2> q7s;
};

enum class SwivelGate {
  kSolved,        // both branch errors written
  kNoTriangle,    // wrist-to-shoulder distance outside the elbow triangle
  kNoCone,        // joint-5 cone condition outside [-1, 1]
  kNearSingular,  // inside a band where the construction is ill-conditioned
};

// Number of q7 samples evaluated together.  Sweep samples are mutually
// independent, and each one's arithmetic is a long serial chain of square roots
// and divisions, so the scalar kernel spends most of its time waiting.
// Evaluating a few at a time lets the hardware overlap those chains and lets
// the SIMD kernels cover the batch with whole vectors: eight lanes fill one
// AVX-512 vector or two AVX2 vectors exactly, and the fine search's default
// scan (2 * n_fine_search = 6 sub-samples at n_fine_search = 3) still fits in
// a single batch -- it just reads only the first `count` lanes.
constexpr int kSwivelLanes = 8;

// How many batches the (cos, sin) recurrence runs before being re-anchored with
// a fresh sincos.  Each rotation costs about an ulp, so this bounds the drift
// well below anything the sweep can notice while keeping the sincos count
// negligible.
constexpr unsigned int kSwivelReanchorBlocks = 16;

// Per-lane outcome of the two acceptance tests.  The sweep and the fine search
// disagree about which of them force a refinement, so the batch records what
// happened and the caller maps it to a gate.
enum : unsigned char {
  kSwivelTriangleOk = 1u,       // elbow triangle closes
  kSwivelNearTriangle = 2u,     // ... but only just: cos(alpha2) > 0.9
  kSwivelConeOk = 4u,           // joint-5 cone condition within [-1, 1]
  kSwivelConeRecoverable = 8u,  // ... outside it, but near enough to refine
  kSwivelNearCone = 16u,        // ... inside it but ill-conditioned
};

struct SwivelBatch {
  // Scaled cosine and sine of the swivel error, indexed [elbow branch][lane]
  // so that each row is contiguous and the lane loop can vectorize.
  double phi_cos[2][kSwivelLanes];
  double phi_sin[2][kSwivelLanes];
  unsigned char flags[kSwivelLanes];
};

constexpr double kTrigDomainLimit = 1.0 + TRIG_DOMAIN_TOL;

// Branch-free counterpart of clamp_trig_roundoff.  Written as two independent
// selects rather than a nested conditional so it if-converts cleanly, and with
// the comparisons oriented so that NaN passes through unclamped -- which is
// what clamp_trig_roundoff does too, since both of its comparisons are false
// for NaN.
static inline double trig_domain_clamp(const double value) {
  const double high = value > 1.0 ? 1.0 : value;
  return high < -1.0 ? -1.0 : high;
}
static inline bool trig_domain_ok(const double value) {
  return !(fabs(value) > kTrigDomainLimit);
}

// Evaluates kSwivelLanes samples of the q7 sweep, given (cos, sin) of
// delta = pi/4 - q7 for each.  The arithmetic is branch-free so lanes cannot
// diverge: a rejected lane still runs it (possibly producing infinities or
// NaNs, which is harmless because nothing reads its results) and is identified
// afterwards from the acceptance tests.
//
// Deliberately split into single-purpose stages.  A square root and a clamp in
// the same loop cannot vectorize: without -fno-math-errno, sqrt keeps a branch
// to a libm call unless the compiler can prove the argument is non-negative,
// and it only manages that when the proof (an fabs or a sum of squares) sits
// lexically inside the sqrt's own argument -- which rules out first clamping
// into a temporary.  Splitting the stages satisfies both, so every stage below
// becomes one or two SIMD instructions instead of four scalar ones.
//
// The fabs guards replace std::max(0.0, x): each argument is non-negative by
// construction (1 - c*c for a c already clamped into [-1, 1] cannot round
// above 1, and l^2 is a squared length), so the guard only absorbs round-off,
// exactly as the max did.
struct SwivelPrefix {
  double p[kSwivelLanes];
  double l2[kSwivelLanes];
  double l[kSwivelLanes];
  double inv_l[kSwivelLanes];
  double cos_actmp[kSwivelLanes];
  double rO_i6[kSwivelLanes];
};

// First stage: how far the wrist centre sits from the shoulder, and whether
// the elbow triangle closes.  Sets the triangle flags and reports whether any
// lane still needs the rest of the construction.
static bool swivel_sample_triangle(const SwivelGeometry& g,
                                   const double* __restrict cos_delta,
                                   const double* __restrict sin_delta,
                                   const bool gated,
                                   SwivelPrefix& __restrict prefix,
                                   SwivelBatch& __restrict out) {
  double cos_raw[kSwivelLanes];
  // i6 is a unit vector and i6 . s6 = 0, so l^2 and r6 . s6 both follow from
  // r_O7S alone.
  for (int lane = 0; lane < kSwivelLanes; ++lane) {
    const double c = cos_delta[lane];
    const double s = sin_delta[lane];
    prefix.rO_i6[lane] = c * g.rO_iE + s * g.rO_jE;
    prefix.p[lane] = c * g.rO_jE - s * g.rO_iE;  // r_O7S . s6 == r6 . s6
    prefix.l2[lane] = g.R2_plus_a7sq - 2.0 * a7 * prefix.rO_i6[lane];
    prefix.l[lane] = sqrt(fabs(prefix.l2[lane]));
  }
  // Elbow triangle: cos(alpha2) = (l^2 + b2^2 - b1^2) / (2*l*b2).
  for (int lane = 0; lane < kSwivelLanes; ++lane) {
    prefix.inv_l[lane] = 1.0 / prefix.l[lane];
    cos_raw[lane] = (prefix.l2[lane] - (b1 * b1 - b2 * b2)) * (0.5 / b2) *
                    prefix.inv_l[lane];
    prefix.cos_actmp[lane] = trig_domain_clamp(cos_raw[lane]);
  }
  bool any_live = false;
  for (int lane = 0; lane < kSwivelLanes; ++lane) {
    unsigned char flags = 0;
    if (trig_domain_ok(cos_raw[lane])) flags |= kSwivelTriangleOk;
    if (prefix.cos_actmp[lane] > 0.9) flags |= kSwivelNearTriangle;
    out.flags[lane] = flags;
    any_live |= (flags & kSwivelTriangleOk) &&
                !(gated && (flags & kSwivelNearTriangle));
  }
  return any_live;
}

// Second stage: the joint-5 cone, the arm assembly and the swivel projections.
static void swivel_sample_assemble(const SwivelGeometry& g,
                                   const double* __restrict cos_delta,
                                   const double* __restrict sin_delta,
                                   const SwivelPrefix& __restrict prefix,
                                   SwivelBatch& __restrict out) {
  const double* const p = prefix.p;
  const double* const l2 = prefix.l2;
  const double* const l = prefix.l;
  const double* const inv_l = prefix.inv_l;
  const double* const cos_actmp = prefix.cos_actmp;
  const double* const rO_i6 = prefix.rO_i6;
  double sin_actmp[kSwivelLanes];
  double sa2[kSwivelLanes], ca2[kSwivelLanes], inv_sa2[kSwivelLanes];
  double n0[kSwivelLanes], n1c[kSwivelLanes], n2c[kSwivelLanes],
      inv_nn[kSwivelLanes];
  double sin_raw[kSwivelLanes], sin_gamma[kSwivelLanes],
      cos_gamma[kSwivelLanes];
  for (int lane = 0; lane < kSwivelLanes; ++lane) {
    sin_actmp[lane] = sqrt(fabs(1.0 - cos_actmp[lane] * cos_actmp[lane]));
  }
  for (int lane = 0; lane < kSwivelLanes; ++lane) {
    sa2[lane] = sin_beta2 * cos_actmp[lane] + cos_beta2 * sin_actmp[lane];
    ca2[lane] = cos_beta2 * cos_actmp[lane] - sin_beta2 * sin_actmp[lane];
  }
  // n = s6 x r6 = c*(j_E x r_O7S) - s*(i_E x r_O7S) + a7*k_E.  Forming it
  // explicitly keeps |n| free of the cancellation that l^2 - (r6.s6)^2 would
  // suffer when s6 nearly aligns with r6.
  for (int lane = 0; lane < kSwivelLanes; ++lane) {
    const double c = cos_delta[lane];
    const double s = sin_delta[lane];
    n0[lane] = c * g.A[0] - s * g.B[0] + g.a7_k_E[0];
    n1c[lane] = c * g.A[1] - s * g.B[1] + g.a7_k_E[1];
    n2c[lane] = c * g.A[2] - s * g.B[2] + g.a7_k_E[2];
  }
  for (int lane = 0; lane < kSwivelLanes; ++lane) {
    inv_nn[lane] = 1.0 / sqrt(n0[lane] * n0[lane] + n1c[lane] * n1c[lane] +
                              n2c[lane] * n2c[lane]);
  }
  // Joint-5 cone: sin(gamma) = -rz*ca2/(ry*sa2) with rz = -(r6.s6)/l and
  // ry = -|n|/l.
  for (int lane = 0; lane < kSwivelLanes; ++lane) {
    inv_sa2[lane] = 1.0 / sa2[lane];
    sin_raw[lane] = -p[lane] * ca2[lane] * inv_nn[lane] * inv_sa2[lane];
    sin_gamma[lane] = trig_domain_clamp(sin_raw[lane]);
  }
  for (int lane = 0; lane < kSwivelLanes; ++lane) {
    cos_gamma[lane] = sqrt(fabs(1.0 - sin_gamma[lane] * sin_gamma[lane]));
  }
  // Assemble r4 = r6_scale*r6 + s5_scale*s5 and project it onto the two
  // constant vectors that carry the swivel error.
  for (int lane = 0; lane < kSwivelLanes; ++lane) {
    const double c = cos_delta[lane];
    const double s = sin_delta[lane];
    const double li = inv_l[lane];
    const double ni = inv_nn[lane];
    const double inverse_s4_norm = li * fabs(inv_sa2[lane]);
    const double l_ca2 = l[lane] * ca2[lane];  // s5 . r6
    const double r6_scale = 1.0 - a5 * inverse_s4_norm;
    const double s5_scale = -d5 + a5 * l_ca2 * inverse_s4_norm;

    // Dot products that do not depend on the cone branch.
    const double rO_r6 = g.R2 - a7 * rO_i6[lane];
    const double r4_r6 = r6_scale * l2[lane] + s5_scale * l_ca2;
    const double r4_s5 = r6_scale * l_ca2 + s5_scale;
    const double orientation_offset = rO_r6 * r4_s5;

    const double c1_i6 = c * g.c1_iE + s * g.c1_jE;
    const double c2_i6 = c * g.c2_iE + s * g.c2_jE;
    const double c1_s6 = c * g.c1_jE - s * g.c1_iE;
    const double c2_s6 = c * g.c2_jE - s * g.c2_iE;
    const double c1_r6 = g.c1_rO - a7 * c1_i6;
    const double c2_r6 = g.c2_rO - a7 * c2_i6;

    // Frame-C components.  k_C = -r6/l, i_C = n/|n|, j_C = k_C x i_C, so
    // w . j_C = -(l*(w . s6) - ((r6.s6)/l)*(w . r6)) / |n|.  For w = r_O7S the
    // i_C term collapses: r_O7S . n = a7 * (r_O7S . k_E), a sweep constant.
    const double p_over_l = p[lane] * li;
    const double c1_iC =
        (g.c1[0] * n0[lane] + g.c1[1] * n1c[lane] + g.c1[2] * n2c[lane]) * ni;
    const double c2_iC =
        (g.c2[0] * n0[lane] + g.c2[1] * n1c[lane] + g.c2[2] * n2c[lane]) * ni;
    const double rO_iC = g.a7_rO_kE * ni;
    const double c1_jC = -ni * (l[lane] * c1_s6 - p_over_l * c1_r6);
    const double c2_jC = -ni * (l[lane] * c2_s6 - p_over_l * c2_r6);
    const double rO_jC = -ni * (l[lane] * p[lane] - p_over_l * rO_r6);
    const double c1_kC = -li * c1_r6;
    const double c2_kC = -li * c2_r6;
    const double rO_kC = -li * rO_r6;

    const double v0 = -sa2[lane] * cos_gamma[lane];
    const double v1 = -sa2[lane] * sin_gamma[lane];
    const double v2 = -ca2[lane];
    const double second = 2.0 * sa2[lane] * cos_gamma[lane];
    const double c1_s5 = v0 * c1_iC + v1 * c1_jC + v2 * c1_kC;
    const double c2_s5 = v0 * c2_iC + v1 * c2_jC + v2 * c2_kC;
    const double rO_s5 = v0 * rO_iC + v1 * rO_jC + v2 * rO_kC;

    // Branch 0 is the cone solution as constructed, branch 1 its reflection
    // across i_C.  Orient the arm-plane normal with the elbow axis: the sign
    // of (r_O7S x r4) . (s5 x r6), which Binet-Cauchy reduces to the form
    // below.
    const double flip0 = rO_s5 * r4_r6 - orientation_offset < 0.0 ? -1.0 : 1.0;
    out.phi_cos[0][lane] = flip0 * (r6_scale * c1_r6 + s5_scale * c1_s5);
    out.phi_sin[0][lane] = flip0 * (r6_scale * c2_r6 + s5_scale * c2_s5);
    const double c1_s5b = c1_s5 + second * c1_iC;
    const double c2_s5b = c2_s5 + second * c2_iC;
    const double rO_s5b = rO_s5 + second * rO_iC;
    const double flip1 = rO_s5b * r4_r6 - orientation_offset < 0.0 ? -1.0 : 1.0;
    out.phi_cos[1][lane] = flip1 * (r6_scale * c1_r6 + s5_scale * c1_s5b);
    out.phi_sin[1][lane] = flip1 * (r6_scale * c2_r6 + s5_scale * c2_s5b);
  }
  for (int lane = 0; lane < kSwivelLanes; ++lane) {
    unsigned char flags = out.flags[lane];
    if (trig_domain_ok(sin_raw[lane])) flags |= kSwivelConeOk;
    if (sin_raw[lane] * sin_raw[lane] < 1.2) flags |= kSwivelConeRecoverable;
    if (sin_gamma[lane] * sin_gamma[lane] > 0.8) flags |= kSwivelNearCone;
    out.flags[lane] = flags;
  }
}

// Advances (cos, sin) of delta through a rotation by the given angle's cosine
// and sine.  delta = pi/4 - q7, so a positive q7 step rotates delta backwards.
static inline void rotate_delta(const double rot_cos, const double rot_sin,
                                double& c, double& s) {
  const double c_next = rot_cos * c - rot_sin * s;
  s = rot_cos * s + rot_sin * c;
  c = c_next;
}

// Per-call table of the rotations a batch needs: lane k sits k steps past the
// batch base, and the base then advances by kSwivelLanes steps.
struct SwivelStepTable {
  double lane_cos[kSwivelLanes];
  double lane_sin[kSwivelLanes];
  double batch_cos;
  double batch_sin;
};

static SwivelStepTable make_step_table(const double q7_step) {
  SwivelStepTable table;
  for (int lane = 0; lane < kSwivelLanes; ++lane) {
    // Stepping q7 forward by lane*q7_step steps delta back by the same amount.
    table.lane_cos[lane] = cos(lane * q7_step);
    table.lane_sin[lane] = -sin(lane * q7_step);
  }
  table.batch_cos = cos(kSwivelLanes * q7_step);
  table.batch_sin = -sin(kSwivelLanes * q7_step);
  return table;
}

static inline void fill_lane_angles(const SwivelStepTable& table,
                                    const double c, const double s,
                                    double* cos_delta, double* sin_delta) {
  for (int lane = 0; lane < kSwivelLanes; ++lane) {
    cos_delta[lane] = table.lane_cos[lane] * c - table.lane_sin[lane] * s;
    sin_delta[lane] = table.lane_cos[lane] * s + table.lane_sin[lane] * c;
  }
}

// Evaluates one batch of samples through the two stages.  The SIMD variants
// land here in a later change; for now this is the scalar path only.
#ifdef NANOGEOFIK_HAVE_AVX2_SWIVEL
// AVX2 versions of the two batch stages.  Each lane executes exactly the same
// operation sequence as the scalar path -- explicit _mm256_* intrinsics, no FMA
// contraction, IEEE divpd/sqrtpd -- so per-lane results are bit-identical to
// the default build; only the four-wide grouping changes.  The trailing pair of
// samples is computed by duplicating lanes 4/5 into the upper half of a second
// vector and storing back only the low 128 bits.
__attribute__((target("avx2"))) static inline __m256d swivel_vclamp(
    const __m256d v) {
  const __m256d one = _mm256_set1_pd(1.0), mone = _mm256_set1_pd(-1.0);
  // Select form (not min/max) so NaN passes through unclamped, like the scalar
  // trig_domain_clamp.
  const __m256d high =
      _mm256_blendv_pd(v, one, _mm256_cmp_pd(v, one, _CMP_GT_OQ));
  return _mm256_blendv_pd(high, mone, _mm256_cmp_pd(high, mone, _CMP_LT_OQ));
}

__attribute__((target("avx2,fma")))
__attribute__((optimize("fp-contract=off"))) static bool
swivel_sample_triangle_avx2(const SwivelGeometry& g, const double* cos_delta,
                            const double* sin_delta, const bool gated,
                            SwivelPrefix& prefix, SwivelBatch& out) {
  const __m256d vR2a7 = _mm256_set1_pd(g.R2_plus_a7sq);
  const __m256d v2a7 = _mm256_set1_pd(2.0 * a7);
  const __m256d vrOiE = _mm256_set1_pd(g.rO_iE),
                vrOjE = _mm256_set1_pd(g.rO_jE);
  const __m256d vC1 = _mm256_set1_pd(b1 * b1 - b2 * b2);
  const __m256d vC2 = _mm256_set1_pd(0.5 / b2);
  const __m256d vpoint9 = _mm256_set1_pd(0.9);
  const __m256d vlimit = _mm256_set1_pd(kTrigDomainLimit);
  const __m256d vsign = _mm256_set1_pd(-0.0);
  bool any_live = false;
  for (int group = 0; group < 2; ++group) {
    const int off = group == 0 ? 0 : 4;
    const __m256d c = _mm256_loadu_pd(cos_delta + off);
    const __m256d s = _mm256_loadu_pd(sin_delta + off);
    const __m256d rO_i6 =
        _mm256_add_pd(_mm256_mul_pd(c, vrOiE), _mm256_mul_pd(s, vrOjE));
    const __m256d p =
        _mm256_sub_pd(_mm256_mul_pd(c, vrOjE), _mm256_mul_pd(s, vrOiE));
    const __m256d l2 = _mm256_sub_pd(vR2a7, _mm256_mul_pd(v2a7, rO_i6));
    const __m256d l = _mm256_sqrt_pd(_mm256_andnot_pd(vsign, l2));
    const __m256d inv_l = _mm256_div_pd(_mm256_set1_pd(1.0), l);
    const __m256d cos_raw =
        _mm256_mul_pd(_mm256_mul_pd(_mm256_sub_pd(l2, vC1), vC2), inv_l);
    const __m256d cos_actmp = swivel_vclamp(cos_raw);
    if (group == 0) {
      _mm256_storeu_pd(prefix.p, p);
      _mm256_storeu_pd(prefix.l2, l2);
      _mm256_storeu_pd(prefix.l, l);
      _mm256_storeu_pd(prefix.inv_l, inv_l);
      _mm256_storeu_pd(prefix.cos_actmp, cos_actmp);
      _mm256_storeu_pd(prefix.rO_i6, rO_i6);
    } else {
      _mm256_storeu_pd(prefix.p + off, p);
      _mm256_storeu_pd(prefix.l2 + off, l2);
      _mm256_storeu_pd(prefix.l + off, l);
      _mm256_storeu_pd(prefix.inv_l + off, inv_l);
      _mm256_storeu_pd(prefix.cos_actmp + off, cos_actmp);
      _mm256_storeu_pd(prefix.rO_i6 + off, rO_i6);
    }
    const __m256d abs_raw = _mm256_andnot_pd(vsign, cos_raw);
    // NaN counts as in-domain, matching trig_domain_ok's !(x > limit).
    const int triangle_ok =
        ~_mm256_movemask_pd(_mm256_cmp_pd(abs_raw, vlimit, _CMP_GT_OQ));
    const int near_tri =
        _mm256_movemask_pd(_mm256_cmp_pd(cos_actmp, vpoint9, _CMP_GT_OQ));
    for (int k = 0; k < 4; ++k) {
      const int lane = off + k;
      unsigned char flags = 0;
      if (triangle_ok & (1 << k)) flags |= kSwivelTriangleOk;
      if (near_tri & (1 << k)) flags |= kSwivelNearTriangle;
      out.flags[lane] = flags;
      any_live |= (flags & kSwivelTriangleOk) &&
                  !(gated && (flags & kSwivelNearTriangle));
    }
  }
  return any_live;
}

__attribute__((target("avx2,fma")))
__attribute__((optimize("fp-contract=off"))) static void
swivel_sample_assemble_avx2(const SwivelGeometry& g, const double* cos_delta,
                            const double* sin_delta, const SwivelPrefix& prefix,
                            SwivelBatch& out) {
  const __m256d vsb2 = _mm256_set1_pd(sin_beta2),
                vcb2 = _mm256_set1_pd(cos_beta2);
  const __m256d vone = _mm256_set1_pd(1.0);
  const __m256d va5 = _mm256_set1_pd(a5), vd5 = _mm256_set1_pd(-d5),
                va7 = _mm256_set1_pd(a7);
  const __m256d vR2 = _mm256_set1_pd(g.R2);
  const __m256d vc1iE = _mm256_set1_pd(g.c1_iE),
                vc1jE = _mm256_set1_pd(g.c1_jE);
  const __m256d vc2iE = _mm256_set1_pd(g.c2_iE),
                vc2jE = _mm256_set1_pd(g.c2_jE);
  const __m256d vc1rO = _mm256_set1_pd(g.c1_rO),
                vc2rO = _mm256_set1_pd(g.c2_rO);
  const __m256d va7rOkE = _mm256_set1_pd(g.a7_rO_kE);
  const __m256d vn0c = _mm256_set1_pd(g.c1[0]), vn1c = _mm256_set1_pd(g.c1[1]),
                vn2c = _mm256_set1_pd(g.c1[2]);
  const __m256d vw0c = _mm256_set1_pd(g.c2[0]), vw1c = _mm256_set1_pd(g.c2[1]),
                vw2c = _mm256_set1_pd(g.c2[2]);
  const __m256d vsign = _mm256_set1_pd(-0.0);
  const __m256d vlimit = _mm256_set1_pd(kTrigDomainLimit);
  const __m256d vmone = _mm256_set1_pd(-1.0), vtwo = _mm256_set1_pd(2.0);
  for (int group = 0; group < 2; ++group) {
    const int off = group == 0 ? 0 : 4;
#define GEOFIK_VLOAD1(arr) (_mm256_loadu_pd((arr) + off))
    const __m256d c = GEOFIK_VLOAD1(cos_delta);
    const __m256d s = GEOFIK_VLOAD1(sin_delta);
    const __m256d cos_actmp = GEOFIK_VLOAD1(prefix.cos_actmp);
    const __m256d p = GEOFIK_VLOAD1(prefix.p);
    const __m256d l2 = GEOFIK_VLOAD1(prefix.l2);
    const __m256d l = GEOFIK_VLOAD1(prefix.l);
    const __m256d li = GEOFIK_VLOAD1(prefix.inv_l);
    const __m256d rO_i6 = GEOFIK_VLOAD1(prefix.rO_i6);
#undef GEOFIK_VLOAD1
    const __m256d sin_actmp = _mm256_sqrt_pd(_mm256_andnot_pd(
        vsign, _mm256_sub_pd(vone, _mm256_mul_pd(cos_actmp, cos_actmp))));
    const __m256d sa2 = _mm256_add_pd(_mm256_mul_pd(vsb2, cos_actmp),
                                      _mm256_mul_pd(vcb2, sin_actmp));
    const __m256d ca2 = _mm256_sub_pd(_mm256_mul_pd(vcb2, cos_actmp),
                                      _mm256_mul_pd(vsb2, sin_actmp));
    // n = c*A - s*B + a7*k_E componentwise.
    const __m256d n0 =
        _mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(c, _mm256_set1_pd(g.A[0])),
                                    _mm256_mul_pd(s, _mm256_set1_pd(g.B[0]))),
                      _mm256_set1_pd(g.a7_k_E[0]));
    const __m256d n1 =
        _mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(c, _mm256_set1_pd(g.A[1])),
                                    _mm256_mul_pd(s, _mm256_set1_pd(g.B[1]))),
                      _mm256_set1_pd(g.a7_k_E[1]));
    const __m256d n2 =
        _mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(c, _mm256_set1_pd(g.A[2])),
                                    _mm256_mul_pd(s, _mm256_set1_pd(g.B[2]))),
                      _mm256_set1_pd(g.a7_k_E[2]));
    const __m256d nn = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(n0, n0), _mm256_mul_pd(n1, n1)),
        _mm256_mul_pd(n2, n2));
    const __m256d ni = _mm256_div_pd(vone, _mm256_sqrt_pd(nn));
    const __m256d inv_sa2 = _mm256_div_pd(vone, sa2);
    const __m256d sin_raw = _mm256_mul_pd(
        _mm256_mul_pd(_mm256_mul_pd(_mm256_xor_pd(vsign, p), ca2), ni),
        inv_sa2);
    const __m256d sin_gamma = swivel_vclamp(sin_raw);
    const __m256d cos_gamma = _mm256_sqrt_pd(_mm256_andnot_pd(
        vsign, _mm256_sub_pd(vone, _mm256_mul_pd(sin_gamma, sin_gamma))));

    const __m256d inverse_s4_norm =
        _mm256_mul_pd(li, _mm256_andnot_pd(vsign, inv_sa2));
    const __m256d l_ca2 = _mm256_mul_pd(l, ca2);
    const __m256d r6_scale =
        _mm256_sub_pd(vone, _mm256_mul_pd(va5, inverse_s4_norm));
    const __m256d s5_scale = _mm256_add_pd(
        vd5, _mm256_mul_pd(_mm256_mul_pd(va5, l_ca2), inverse_s4_norm));
    const __m256d rO_r6 = _mm256_sub_pd(vR2, _mm256_mul_pd(va7, rO_i6));
    const __m256d r4_r6 = _mm256_add_pd(_mm256_mul_pd(r6_scale, l2),
                                        _mm256_mul_pd(s5_scale, l_ca2));
    const __m256d r4_s5 =
        _mm256_add_pd(_mm256_mul_pd(r6_scale, l_ca2), s5_scale);
    const __m256d orientation_offset = _mm256_mul_pd(rO_r6, r4_s5);
    const __m256d c1_i6 =
        _mm256_add_pd(_mm256_mul_pd(c, vc1iE), _mm256_mul_pd(s, vc1jE));
    const __m256d c2_i6 =
        _mm256_add_pd(_mm256_mul_pd(c, vc2iE), _mm256_mul_pd(s, vc2jE));
    const __m256d c1_s6 =
        _mm256_sub_pd(_mm256_mul_pd(c, vc1jE), _mm256_mul_pd(s, vc1iE));
    const __m256d c2_s6 =
        _mm256_sub_pd(_mm256_mul_pd(c, vc2jE), _mm256_mul_pd(s, vc2iE));
    const __m256d c1_r6 = _mm256_sub_pd(vc1rO, _mm256_mul_pd(va7, c1_i6));
    const __m256d c2_r6 = _mm256_sub_pd(vc2rO, _mm256_mul_pd(va7, c2_i6));
    const __m256d p_over_l = _mm256_mul_pd(p, li);
    const __m256d c1_iC = _mm256_mul_pd(
        _mm256_add_pd(
            _mm256_add_pd(_mm256_mul_pd(vn0c, n0), _mm256_mul_pd(vn1c, n1)),
            _mm256_mul_pd(vn2c, n2)),
        ni);
    const __m256d c2_iC = _mm256_mul_pd(
        _mm256_add_pd(
            _mm256_add_pd(_mm256_mul_pd(vw0c, n0), _mm256_mul_pd(vw1c, n1)),
            _mm256_mul_pd(vw2c, n2)),
        ni);
    const __m256d rO_iC = _mm256_mul_pd(va7rOkE, ni);
    const __m256d c2_jC = _mm256_mul_pd(
        _mm256_xor_pd(
            vsign,
            _mm256_mul_pd(ni, _mm256_sub_pd(_mm256_mul_pd(l, c2_s6),
                                            _mm256_mul_pd(p_over_l, c2_r6)))),
        vone);
    const __m256d c1_jC = _mm256_mul_pd(
        _mm256_xor_pd(
            vsign,
            _mm256_mul_pd(ni, _mm256_sub_pd(_mm256_mul_pd(l, c1_s6),
                                            _mm256_mul_pd(p_over_l, c1_r6)))),
        vone);
    const __m256d rO_jC = _mm256_mul_pd(
        _mm256_xor_pd(
            vsign,
            _mm256_mul_pd(ni, _mm256_sub_pd(_mm256_mul_pd(l, p),
                                            _mm256_mul_pd(p_over_l, rO_r6)))),
        vone);
    const __m256d c1_kC =
        _mm256_mul_pd(_mm256_xor_pd(vsign, _mm256_mul_pd(li, c1_r6)), vone);
    const __m256d c2_kC =
        _mm256_mul_pd(_mm256_xor_pd(vsign, _mm256_mul_pd(li, c2_r6)), vone);
    const __m256d rO_kC =
        _mm256_mul_pd(_mm256_xor_pd(vsign, _mm256_mul_pd(li, rO_r6)), vone);
    const __m256d v0 =
        _mm256_mul_pd(_mm256_set1_pd(-1.0), _mm256_mul_pd(sa2, cos_gamma));
    const __m256d v1 =
        _mm256_mul_pd(_mm256_set1_pd(-1.0), _mm256_mul_pd(sa2, sin_gamma));
    const __m256d v2 = _mm256_mul_pd(_mm256_set1_pd(-1.0), ca2);
    const __m256d second = _mm256_mul_pd(vtwo, _mm256_mul_pd(sa2, cos_gamma));
    const __m256d c1_s5 = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(v0, c1_iC), _mm256_mul_pd(v1, c1_jC)),
        _mm256_mul_pd(v2, c1_kC));
    const __m256d c2_s5 = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(v0, c2_iC), _mm256_mul_pd(v1, c2_jC)),
        _mm256_mul_pd(v2, c2_kC));
    const __m256d rO_s5 = _mm256_add_pd(
        _mm256_add_pd(_mm256_mul_pd(v0, rO_iC), _mm256_mul_pd(v1, rO_jC)),
        _mm256_mul_pd(v2, rO_kC));
    const __m256d t0 =
        _mm256_sub_pd(_mm256_mul_pd(rO_s5, r4_r6), orientation_offset);
    const __m256d flip0 = _mm256_blendv_pd(
        vone, vmone, _mm256_cmp_pd(t0, _mm256_setzero_pd(), _CMP_LT_OQ));
    const __m256d phi_cos0 =
        _mm256_mul_pd(flip0, _mm256_add_pd(_mm256_mul_pd(r6_scale, c1_r6),
                                           _mm256_mul_pd(s5_scale, c1_s5)));
    const __m256d phi_sin0 =
        _mm256_mul_pd(flip0, _mm256_add_pd(_mm256_mul_pd(r6_scale, c2_r6),
                                           _mm256_mul_pd(s5_scale, c2_s5)));
    const __m256d c1_s5b = _mm256_add_pd(c1_s5, _mm256_mul_pd(second, c1_iC));
    const __m256d c2_s5b = _mm256_add_pd(c2_s5, _mm256_mul_pd(second, c2_iC));
    const __m256d rO_s5b = _mm256_add_pd(rO_s5, _mm256_mul_pd(second, rO_iC));
    const __m256d t1 =
        _mm256_sub_pd(_mm256_mul_pd(rO_s5b, r4_r6), orientation_offset);
    const __m256d flip1 = _mm256_blendv_pd(
        vone, vmone, _mm256_cmp_pd(t1, _mm256_setzero_pd(), _CMP_LT_OQ));
    const __m256d phi_cos1 =
        _mm256_mul_pd(flip1, _mm256_add_pd(_mm256_mul_pd(r6_scale, c1_r6),
                                           _mm256_mul_pd(s5_scale, c1_s5b)));
    const __m256d phi_sin1 =
        _mm256_mul_pd(flip1, _mm256_add_pd(_mm256_mul_pd(r6_scale, c2_r6),
                                           _mm256_mul_pd(s5_scale, c2_s5b)));
    _mm256_storeu_pd(out.phi_cos[0] + off, phi_cos0);
    _mm256_storeu_pd(out.phi_sin[0] + off, phi_sin0);
    _mm256_storeu_pd(out.phi_cos[1] + off, phi_cos1);
    _mm256_storeu_pd(out.phi_sin[1] + off, phi_sin1);
    const __m256d abs_sin_raw = _mm256_andnot_pd(vsign, sin_raw);
    // NaN counts as in-domain, matching trig_domain_ok's !(x > limit).
    const int cone_ok =
        ~_mm256_movemask_pd(_mm256_cmp_pd(abs_sin_raw, vlimit, _CMP_GT_OQ));
    const __m256d sr2 = _mm256_mul_pd(sin_raw, sin_raw);
    const int recoverable =
        _mm256_movemask_pd(_mm256_cmp_pd(sr2, _mm256_set1_pd(1.2), _CMP_LT_OQ));
    const __m256d sg2 = _mm256_mul_pd(sin_gamma, sin_gamma);
    const int near_cone =
        _mm256_movemask_pd(_mm256_cmp_pd(sg2, _mm256_set1_pd(0.8), _CMP_GT_OQ));
    for (int k = 0; k < 4; ++k) {
      unsigned char flags = out.flags[off + k];
      if (cone_ok & (1 << k)) flags |= kSwivelConeOk;
      if (recoverable & (1 << k)) flags |= kSwivelConeRecoverable;
      if (near_cone & (1 << k)) flags |= kSwivelNearCone;
      out.flags[off + k] = flags;
    }
  }
}

#ifdef __AVX512F__
__attribute__((target("avx512f"))) static inline __m512d swivel_vclamp_512(
    const __m512d v) {
  const __m512d one = _mm512_set1_pd(1.0), mone = _mm512_set1_pd(-1.0);
  const __mmask8 high_mask = _mm512_cmp_pd_mask(v, one, _CMP_GT_OQ);
  const __m512d high = _mm512_mask_blend_pd(high_mask, v, one);
  const __mmask8 low_mask = _mm512_cmp_pd_mask(high, mone, _CMP_LT_OQ);
  return _mm512_mask_blend_pd(low_mask, high, mone);
}

// AVX-512 versions of the two batch stages: same per-lane operation sequence
// as the scalar and AVX2 paths -- explicit intrinsics, no FMA contraction,
// IEEE divpd/sqrtpd -- so results are bit-identical; only the grouping
// (one full 8-lane vector) changes.

__attribute__((target("avx512f"))) static bool swivel_sample_triangle_avx512(
    const SwivelGeometry& g, const double* cos_delta, const double* sin_delta,
    const bool gated, SwivelPrefix& prefix, SwivelBatch& out) {
  const __m512d vR2a7 = _mm512_set1_pd(g.R2_plus_a7sq);
  const __m512d v2a7 = _mm512_set1_pd(2.0 * a7);
  const __m512d vrOiE = _mm512_set1_pd(g.rO_iE),
                vrOjE = _mm512_set1_pd(g.rO_jE);
  const __m512d vC1 = _mm512_set1_pd(b1 * b1 - b2 * b2);
  const __m512d vC2 = _mm512_set1_pd(0.5 / b2);
  const __m512d vpoint9 = _mm512_set1_pd(0.9);
  const __m512d vlimit = _mm512_set1_pd(kTrigDomainLimit);
  const __m512d vsign = _mm512_set1_pd(-0.0);
  const __m512d c = _mm512_loadu_pd(cos_delta);
  const __m512d s = _mm512_loadu_pd(sin_delta);
  const __m512d rO_i6 =
      _mm512_add_pd(_mm512_mul_pd(c, vrOiE), _mm512_mul_pd(s, vrOjE));
  const __m512d p =
      _mm512_sub_pd(_mm512_mul_pd(c, vrOjE), _mm512_mul_pd(s, vrOiE));
  const __m512d l2 = _mm512_sub_pd(vR2a7, _mm512_mul_pd(v2a7, rO_i6));
  const __m512d l = _mm512_sqrt_pd(_mm512_andnot_pd(vsign, l2));
  const __m512d inv_l = _mm512_div_pd(_mm512_set1_pd(1.0), l);

  const __m512d cos_raw =
      _mm512_mul_pd(_mm512_mul_pd(_mm512_sub_pd(l2, vC1), vC2), inv_l);
  const __m512d cos_actmp = swivel_vclamp_512(cos_raw);
  _mm512_storeu_pd(prefix.p, p);
  _mm512_storeu_pd(prefix.l2, l2);
  _mm512_storeu_pd(prefix.l, l);
  _mm512_storeu_pd(prefix.inv_l, inv_l);
  _mm512_storeu_pd(prefix.cos_actmp, cos_actmp);
  _mm512_storeu_pd(prefix.rO_i6, rO_i6);
  const __m512d abs_raw = _mm512_andnot_pd(vsign, cos_raw);
  // NaN counts as in-domain, matching trig_domain_ok's !(x > limit).
  const __mmask8 triangle_ok =
      __mmask8(~_mm512_cmp_pd_mask(abs_raw, vlimit, _CMP_GT_OQ));
  const __mmask8 near_tri = _mm512_cmp_pd_mask(cos_actmp, vpoint9, _CMP_GT_OQ);
  bool any_live = false;
  for (int k = 0; k < kSwivelLanes; ++k) {
    unsigned char flags = 0;
    if (triangle_ok & (1 << k)) flags |= kSwivelTriangleOk;
    if (near_tri & (1 << k)) flags |= kSwivelNearTriangle;
    out.flags[k] = flags;
    any_live |= (flags & kSwivelTriangleOk) &&
                !(gated && (flags & kSwivelNearTriangle));
  }
  return any_live;
}

__attribute__((target("avx512f")))
__attribute__((optimize("fp-contract=off"))) static void
swivel_sample_assemble_avx512(const SwivelGeometry& g, const double* cos_delta,
                              const double* sin_delta,
                              const SwivelPrefix& prefix, SwivelBatch& out) {
  const __m512d vsb2 = _mm512_set1_pd(sin_beta2),
                vcb2 = _mm512_set1_pd(cos_beta2);
  const __m512d vone = _mm512_set1_pd(1.0);
  const __m512d va5 = _mm512_set1_pd(a5), vd5 = _mm512_set1_pd(-d5),
                va7 = _mm512_set1_pd(a7);
  const __m512d vR2 = _mm512_set1_pd(g.R2);
  const __m512d vc1iE = _mm512_set1_pd(g.c1_iE),
                vc1jE = _mm512_set1_pd(g.c1_jE);
  const __m512d vc2iE = _mm512_set1_pd(g.c2_iE),
                vc2jE = _mm512_set1_pd(g.c2_jE);
  const __m512d vc1rO = _mm512_set1_pd(g.c1_rO),
                vc2rO = _mm512_set1_pd(g.c2_rO);
  const __m512d va7rOkE = _mm512_set1_pd(g.a7_rO_kE);
  const __m512d vn0c = _mm512_set1_pd(g.c1[0]), vn1c = _mm512_set1_pd(g.c1[1]),
                vn2c = _mm512_set1_pd(g.c1[2]);
  const __m512d vw0c = _mm512_set1_pd(g.c2[0]), vw1c = _mm512_set1_pd(g.c2[1]),
                vw2c = _mm512_set1_pd(g.c2[2]);
  const __m512d vsign = _mm512_set1_pd(-0.0);
  const __m512d vlimit = _mm512_set1_pd(kTrigDomainLimit);
  const __m512d vmone = _mm512_set1_pd(-1.0), vtwo = _mm512_set1_pd(2.0);
  const __m512d c = _mm512_loadu_pd(cos_delta);
  const __m512d s = _mm512_loadu_pd(sin_delta);
  const __m512d cos_actmp = _mm512_loadu_pd(prefix.cos_actmp);
  const __m512d p = _mm512_loadu_pd(prefix.p);
  const __m512d l2 = _mm512_loadu_pd(prefix.l2);
  const __m512d l = _mm512_loadu_pd(prefix.l);
  const __m512d li = _mm512_loadu_pd(prefix.inv_l);
  const __m512d rO_i6 = _mm512_loadu_pd(prefix.rO_i6);
  const __m512d sin_actmp = _mm512_sqrt_pd(_mm512_andnot_pd(
      vsign, _mm512_sub_pd(vone, _mm512_mul_pd(cos_actmp, cos_actmp))));
  const __m512d sa2 = _mm512_add_pd(_mm512_mul_pd(vsb2, cos_actmp),
                                    _mm512_mul_pd(vcb2, sin_actmp));
  const __m512d ca2 = _mm512_sub_pd(_mm512_mul_pd(vcb2, cos_actmp),
                                    _mm512_mul_pd(vsb2, sin_actmp));
  // n = c*A - s*B + a7*k_E componentwise.
  const __m512d n0 =
      _mm512_add_pd(_mm512_sub_pd(_mm512_mul_pd(c, _mm512_set1_pd(g.A[0])),
                                  _mm512_mul_pd(s, _mm512_set1_pd(g.B[0]))),
                    _mm512_set1_pd(g.a7_k_E[0]));
  const __m512d n1 =
      _mm512_add_pd(_mm512_sub_pd(_mm512_mul_pd(c, _mm512_set1_pd(g.A[1])),
                                  _mm512_mul_pd(s, _mm512_set1_pd(g.B[1]))),
                    _mm512_set1_pd(g.a7_k_E[1]));
  const __m512d n2 =
      _mm512_add_pd(_mm512_sub_pd(_mm512_mul_pd(c, _mm512_set1_pd(g.A[2])),
                                  _mm512_mul_pd(s, _mm512_set1_pd(g.B[2]))),
                    _mm512_set1_pd(g.a7_k_E[2]));
  const __m512d nn =
      _mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(n0, n0), _mm512_mul_pd(n1, n1)),
                    _mm512_mul_pd(n2, n2));
  const __m512d ni = _mm512_div_pd(vone, _mm512_sqrt_pd(nn));
  const __m512d inv_sa2 = _mm512_div_pd(vone, sa2);
  const __m512d sin_raw = _mm512_mul_pd(
      _mm512_mul_pd(_mm512_mul_pd(_mm512_xor_pd(vsign, p), ca2), ni), inv_sa2);
  const __m512d sin_gamma = swivel_vclamp_512(sin_raw);
  const __m512d cos_gamma = _mm512_sqrt_pd(_mm512_andnot_pd(
      vsign, _mm512_sub_pd(vone, _mm512_mul_pd(sin_gamma, sin_gamma))));

  const __m512d inverse_s4_norm =
      _mm512_mul_pd(li, _mm512_andnot_pd(vsign, inv_sa2));
  const __m512d l_ca2 = _mm512_mul_pd(l, ca2);
  const __m512d r6_scale =
      _mm512_sub_pd(vone, _mm512_mul_pd(va5, inverse_s4_norm));
  const __m512d s5_scale = _mm512_add_pd(
      vd5, _mm512_mul_pd(_mm512_mul_pd(va5, l_ca2), inverse_s4_norm));
  const __m512d rO_r6 = _mm512_sub_pd(vR2, _mm512_mul_pd(va7, rO_i6));
  const __m512d r4_r6 = _mm512_add_pd(_mm512_mul_pd(r6_scale, l2),
                                      _mm512_mul_pd(s5_scale, l_ca2));
  const __m512d r4_s5 = _mm512_add_pd(_mm512_mul_pd(r6_scale, l_ca2), s5_scale);
  const __m512d orientation_offset = _mm512_mul_pd(rO_r6, r4_s5);
  const __m512d c1_i6 =
      _mm512_add_pd(_mm512_mul_pd(c, vc1iE), _mm512_mul_pd(s, vc1jE));
  const __m512d c2_i6 =
      _mm512_add_pd(_mm512_mul_pd(c, vc2iE), _mm512_mul_pd(s, vc2jE));
  const __m512d c1_s6 =
      _mm512_sub_pd(_mm512_mul_pd(c, vc1jE), _mm512_mul_pd(s, vc1iE));
  const __m512d c2_s6 =
      _mm512_sub_pd(_mm512_mul_pd(c, vc2jE), _mm512_mul_pd(s, vc2iE));
  const __m512d c1_r6 = _mm512_sub_pd(vc1rO, _mm512_mul_pd(va7, c1_i6));
  const __m512d c2_r6 = _mm512_sub_pd(vc2rO, _mm512_mul_pd(va7, c2_i6));
  const __m512d p_over_l = _mm512_mul_pd(p, li);
  const __m512d c1_iC = _mm512_mul_pd(
      _mm512_add_pd(
          _mm512_add_pd(_mm512_mul_pd(vn0c, n0), _mm512_mul_pd(vn1c, n1)),
          _mm512_mul_pd(vn2c, n2)),
      ni);
  const __m512d c2_iC = _mm512_mul_pd(
      _mm512_add_pd(
          _mm512_add_pd(_mm512_mul_pd(vw0c, n0), _mm512_mul_pd(vw1c, n1)),
          _mm512_mul_pd(vw2c, n2)),
      ni);
  const __m512d rO_iC = _mm512_mul_pd(va7rOkE, ni);
  const __m512d c2_jC = _mm512_xor_pd(
      vsign, _mm512_mul_pd(ni, _mm512_sub_pd(_mm512_mul_pd(l, c2_s6),
                                             _mm512_mul_pd(p_over_l, c2_r6))));
  const __m512d c1_jC = _mm512_xor_pd(
      vsign, _mm512_mul_pd(ni, _mm512_sub_pd(_mm512_mul_pd(l, c1_s6),
                                             _mm512_mul_pd(p_over_l, c1_r6))));
  const __m512d rO_jC = _mm512_xor_pd(
      vsign, _mm512_mul_pd(ni, _mm512_sub_pd(_mm512_mul_pd(l, p),
                                             _mm512_mul_pd(p_over_l, rO_r6))));
  const __m512d c1_kC = _mm512_xor_pd(vsign, _mm512_mul_pd(li, c1_r6));
  const __m512d c2_kC = _mm512_xor_pd(vsign, _mm512_mul_pd(li, c2_r6));
  const __m512d rO_kC = _mm512_xor_pd(vsign, _mm512_mul_pd(li, rO_r6));
  const __m512d v0 =
      _mm512_mul_pd(_mm512_set1_pd(-1.0), _mm512_mul_pd(sa2, cos_gamma));
  const __m512d v1 =
      _mm512_mul_pd(_mm512_set1_pd(-1.0), _mm512_mul_pd(sa2, sin_gamma));
  const __m512d v2 = _mm512_mul_pd(_mm512_set1_pd(-1.0), ca2);
  const __m512d second = _mm512_mul_pd(vtwo, _mm512_mul_pd(sa2, cos_gamma));
  const __m512d c1_s5 = _mm512_add_pd(
      _mm512_add_pd(_mm512_mul_pd(v0, c1_iC), _mm512_mul_pd(v1, c1_jC)),
      _mm512_mul_pd(v2, c1_kC));
  const __m512d c2_s5 = _mm512_add_pd(
      _mm512_add_pd(_mm512_mul_pd(v0, c2_iC), _mm512_mul_pd(v1, c2_jC)),
      _mm512_mul_pd(v2, c2_kC));
  const __m512d rO_s5 = _mm512_add_pd(
      _mm512_add_pd(_mm512_mul_pd(v0, rO_iC), _mm512_mul_pd(v1, rO_jC)),
      _mm512_mul_pd(v2, rO_kC));
  const __m512d t0 =
      _mm512_sub_pd(_mm512_mul_pd(rO_s5, r4_r6), orientation_offset);
  const __m512d flip0 = _mm512_mask_blend_pd(
      _mm512_cmp_pd_mask(t0, _mm512_setzero_pd(), _CMP_LT_OQ), vone, vmone);
  const __m512d phi_cos0 =
      _mm512_mul_pd(flip0, _mm512_add_pd(_mm512_mul_pd(r6_scale, c1_r6),
                                         _mm512_mul_pd(s5_scale, c1_s5)));
  const __m512d phi_sin0 =
      _mm512_mul_pd(flip0, _mm512_add_pd(_mm512_mul_pd(r6_scale, c2_r6),
                                         _mm512_mul_pd(s5_scale, c2_s5)));
  const __m512d c1_s5b = _mm512_add_pd(c1_s5, _mm512_mul_pd(second, c1_iC));
  const __m512d c2_s5b = _mm512_add_pd(c2_s5, _mm512_mul_pd(second, c2_iC));
  const __m512d rO_s5b = _mm512_add_pd(rO_s5, _mm512_mul_pd(second, rO_iC));
  const __m512d t1 =
      _mm512_sub_pd(_mm512_mul_pd(rO_s5b, r4_r6), orientation_offset);
  const __m512d flip1 = _mm512_mask_blend_pd(
      _mm512_cmp_pd_mask(t1, _mm512_setzero_pd(), _CMP_LT_OQ), vone, vmone);
  const __m512d phi_cos1 =
      _mm512_mul_pd(flip1, _mm512_add_pd(_mm512_mul_pd(r6_scale, c1_r6),
                                         _mm512_mul_pd(s5_scale, c1_s5b)));
  const __m512d phi_sin1 =
      _mm512_mul_pd(flip1, _mm512_add_pd(_mm512_mul_pd(r6_scale, c2_r6),
                                         _mm512_mul_pd(s5_scale, c2_s5b)));
  _mm512_storeu_pd(out.phi_cos[0], phi_cos0);
  _mm512_storeu_pd(out.phi_sin[0], phi_sin0);
  _mm512_storeu_pd(out.phi_cos[1], phi_cos1);
  _mm512_storeu_pd(out.phi_sin[1], phi_sin1);
  const __m512d abs_sin_raw = _mm512_andnot_pd(vsign, sin_raw);
  // NaN counts as in-domain, matching trig_domain_ok's !(x > limit).
  const __mmask8 cone_ok =
      __mmask8(~_mm512_cmp_pd_mask(abs_sin_raw, vlimit, _CMP_GT_OQ));
  const __m512d sr2 = _mm512_mul_pd(sin_raw, sin_raw);
  const __mmask8 recoverable =
      _mm512_cmp_pd_mask(sr2, _mm512_set1_pd(1.2), _CMP_LT_OQ);
  const __m512d sg2 = _mm512_mul_pd(sin_gamma, sin_gamma);
  const __mmask8 near_cone =
      _mm512_cmp_pd_mask(sg2, _mm512_set1_pd(0.8), _CMP_GT_OQ);
  for (int k = 0; k < kSwivelLanes; ++k) {
    unsigned char flags = out.flags[k];
    if (cone_ok & (1 << k)) flags |= kSwivelConeOk;
    if (recoverable & (1 << k)) flags |= kSwivelConeRecoverable;
    if (near_cone & (1 << k)) flags |= kSwivelNearCone;
    out.flags[k] = flags;
  }
}
#endif  // __AVX512F__
#endif
static inline void swivel_sample_batch(const SwivelGeometry& g,
                                       const double* __restrict cos_delta,
                                       const double* __restrict sin_delta,
                                       const bool gated,
                                       SwivelBatch& __restrict out) {
#ifdef NANOGEOFIK_HAVE_AVX2_SWIVEL
#if defined(__x86_64__) && defined(__AVX512F__)
  static const bool has_avx512 = __builtin_cpu_supports("avx512f");
  if (has_avx512) {
    SwivelPrefix prefix;
    if (swivel_sample_triangle_avx512(g, cos_delta, sin_delta, gated, prefix,
                                      out))
      swivel_sample_assemble_avx512(g, cos_delta, sin_delta, prefix, out);
    return;
  }
#endif
  static const bool has_avx2 =
      __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  if (has_avx2) {
    SwivelPrefix prefix;
    if (swivel_sample_triangle_avx2(g, cos_delta, sin_delta, gated, prefix,
                                    out))
      swivel_sample_assemble_avx2(g, cos_delta, sin_delta, prefix, out);
    return;
  }
#endif
  SwivelPrefix prefix;
  if (swivel_sample_triangle(g, cos_delta, sin_delta, gated, prefix, out))
    swivel_sample_assemble(g, cos_delta, sin_delta, prefix, out);
}

// Maps a lane's flags to the sweep's decision.  `gated` is false for the fine
// search itself (and when the caller disabled refinement), where an
// ill-conditioned sample is simply evaluated as-is.
static inline SwivelGate swivel_gate(const unsigned char flags,
                                     const bool gated) {
  if (!(flags & kSwivelTriangleOk)) return SwivelGate::kNoTriangle;
  if (gated && (flags & kSwivelNearTriangle)) return SwivelGate::kNearSingular;
  if (!(flags & kSwivelConeOk))
    return gated && (flags & kSwivelConeRecoverable) ? SwivelGate::kNearSingular
                                                     : SwivelGate::kNoCone;
  if (gated && (flags & kSwivelNearCone)) return SwivelGate::kNearSingular;
  return SwivelGate::kSolved;
}

struct SwivelMinima {
  array<array<unsigned int, 2>, 4> values;
  unsigned int count = 0;
};

struct SwivelSweep {
  array<array<double, 2>, MAX_N_POINTS> signed_errors;
  array<array<double, 2>, MAX_N_POINTS> q7s;
  SwivelMinima minima;
};

static SwivelMinima find_swivel_minima(
    const array<array<double, 2>, MAX_N_POINTS>& signed_errors,
    const unsigned int n_points) {
  SwivelMinima result;
  // Process branch 0 before branch 1 to preserve GeoFIK's public solution
  // order.
  for (unsigned int branch = 0; branch < 2; ++branch) {
    bool in_run = false;
    unsigned int minimum = 0;
    for (unsigned int i = 0; i < n_points; ++i) {
      if (fabs(signed_errors[i][branch]) < ERR_THRESH) {
        if (!in_run) {
          minimum = i;
          in_run = true;
        } else if (fabs(signed_errors[i][branch]) <
                   fabs(signed_errors[minimum][branch])) {
          minimum = i;
        }
      } else if (in_run) {
        if (result.count < result.values.size())
          result.values[result.count++] = {minimum, branch};
        in_run = false;
      }
    }
    if (in_run && result.count < result.values.size())
      result.values[result.count++] = {minimum, branch};
  }
  return result;
}
static SwivelSample swivel_fine_search(const SwivelGeometry& g,
                                       const double q7m, const double c_mid,
                                       const double s_mid, const double q7step,
                                       const double fine_step,
                                       const SwivelStepTable& fine_table,
                                       const double half_step_cos,
                                       const double half_step_sin) {
  array<double, 2> best_q7s = {q7m, q7m};
  array<double, 2> min_errs = {1e15, 1e15};
  array<double, 2> min_signed_errs = {1e15, 1e15};
  const double q7_end = q7m + q7step / 2;
  // delta = pi/4 - q7, so stepping q7 back by q7step/2 steps delta forward.
  double c = c_mid * half_step_cos - s_mid * half_step_sin;
  double s = s_mid * half_step_cos + c_mid * half_step_sin;
  double q7j = q7m - q7step / 2;
  double cos_delta[kSwivelLanes], sin_delta[kSwivelLanes];
  double q7_lane[kSwivelLanes];
  SwivelBatch batch;
  while (q7j < q7_end) {
    fill_lane_angles(fine_table, c, s, cos_delta, sin_delta);
    int count = 0;
    while (count < kSwivelLanes && q7j < q7_end) {
      q7_lane[count++] = q7j;
      q7j += fine_step;
    }
    swivel_sample_batch(g, cos_delta, sin_delta, false, batch);
    for (int lane = 0; lane < count; ++lane) {
      if (swivel_gate(batch.flags[lane], false) != SwivelGate::kSolved)
        continue;
      for (int branch = 0; branch < 2; ++branch) {
        const double signed_err = swivel_error_from_scaled(
            g.theta_sin, g.theta_cos, batch.phi_cos[branch][lane],
            batch.phi_sin[branch][lane]);
        const double err = fabs(signed_err);
        if (err < min_errs[branch]) {
          best_q7s[branch] = q7_lane[lane];
          min_errs[branch] = err;
          min_signed_errs[branch] = signed_err;
        }
      }
    }
    rotate_delta(fine_table.batch_cos, fine_table.batch_sin, c, s);
  }
  return SwivelSample{min_signed_errs, best_q7s};
}

// One q7 sweep: sample the range, refining any sample that lands in a
// near-singular band, and return the bracketing minima.  Shared by the
// joint-angle and Jacobian entry points, which differ only in how they turn
// the located q7 values into solutions.
static void run_swivel_sweep(const SwivelGeometry& g,
                             const unsigned int n_points,
                             const unsigned int n_fine_search,
                             const SolverTuning& tuning, SwivelSweep& sweep) {
  const JointLimits& limits = resolve_limits(tuning);
  const double q7_low = limits.lower[6];
  const double q7_step = (limits.upper[6] - q7_low) / (n_points - 1);
  const SwivelStepTable sweep_table = make_step_table(q7_step);
  // n_fine_search == 0 disables refinement, so the fine table is never used.
  const double fine_step =
      n_fine_search == 0 ? 0.0 : q7_step / (2 * n_fine_search);
  const SwivelStepTable fine_table = make_step_table(fine_step);
  const double half_step_cos = cos(0.5 * q7_step);
  const double half_step_sin = sin(0.5 * q7_step);
  const bool gated = n_fine_search > 0;
  double c = cos(PI / 4.0 - q7_low);
  double s = sin(PI / 4.0 - q7_low);
  double cos_delta[kSwivelLanes], sin_delta[kSwivelLanes];
  SwivelBatch batch;
  unsigned int block = 0;
  for (unsigned int base = 0; base < n_points; base += kSwivelLanes, ++block) {
    // Re-anchor periodically so the (cos, sin) recurrence cannot accumulate
    // meaningful round-off over the maximum 1000-point sweep.  Counting blocks
    // rather than samples keeps the bound independent of the lane count.
    if (block != 0 && block % kSwivelReanchorBlocks == 0) {
      const double delta = PI / 4.0 - (q7_low + base * q7_step);
      c = cos(delta);
      s = sin(delta);
    }
    fill_lane_angles(sweep_table, c, s, cos_delta, sin_delta);
    swivel_sample_batch(g, cos_delta, sin_delta, gated, batch);
    const unsigned int count = n_points - base < kSwivelLanes
                                   ? n_points - base
                                   : unsigned(kSwivelLanes);
    for (unsigned int lane = 0; lane < count; ++lane) {
      const unsigned int i = base + lane;
      const double q7 = q7_low + i * q7_step;
      switch (swivel_gate(batch.flags[lane], gated)) {
        case SwivelGate::kSolved:
          for (int branch = 0; branch < 2; ++branch)
            sweep.signed_errors[i][branch] = swivel_error_from_scaled(
                g.theta_sin, g.theta_cos, batch.phi_cos[branch][lane],
                batch.phi_sin[branch][lane]);
          sweep.q7s[i] = {q7, q7};
          break;
        case SwivelGate::kNearSingular: {
          const SwivelSample refined = swivel_fine_search(
              g, q7, cos_delta[lane], sin_delta[lane], q7_step, fine_step,
              fine_table, half_step_cos, half_step_sin);
          sweep.signed_errors[i] = refined.signed_errors;
          sweep.q7s[i] = refined.q7s;
          break;
        }
        case SwivelGate::kNoTriangle:
          sweep.signed_errors[i] = {1e10, 1e10};
          sweep.q7s[i] = {q7, q7};
          break;
        default:
          sweep.signed_errors[i] = {1e15, 1e15};
          sweep.q7s[i] = {q7, q7};
          break;
      }
    }
    rotate_delta(sweep_table.batch_cos, sweep_table.batch_sin, c, s);
  }
  sweep.minima = find_swivel_minima(sweep.signed_errors, n_points);
}

// Refine a sample that fell in a near-singular band by scanning the cell
// around it and keeping, per elbow branch, the sub-sample with the smallest
// error.
static void franka_ik_q7_one_sol(const double q7, const array<double, 3>& i_E_O,
                                 const array<double, 3>& j_E_O,
                                 const array<double, 3>& k_E_O,
                                 array<double, 3>& i_6_O,
                                 const array<double, 3>& r_O7S_O,
                                 const unsigned int branch,
                                 array<array<double, 7>, 8>& qsols,
                                 unsigned int ind, const SolverTuning& tuning) {
  const double q1_sing = tuning.q1_sing;
  // returns the two solution related to one single branch of the IK with q7 as
  // free variable. The results are stored in qsols[s*ind] and qsols[2*ind+1]
  array<double, 3> s6;
  wrist_axes_from_q7(i_E_O, j_E_O, q7, i_6_O, s6);
  array<double, 3> r6 = {r_O7S_O[0] - a7 * i_6_O[0], r_O7S_O[1] - a7 * i_6_O[1],
                         r_O7S_O[2] - a7 * i_6_O[2]};
  double l = Norm(r6);
  double tmp = (b1 * b1 - l * l - b2 * b2) / (-2 * l * b2);
  if (!clamp_trig_roundoff(tmp)) {
    fill(qsols[2 * ind].begin(), qsols[2 * ind].end(), NAN);
    fill(qsols[2 * ind + 1].begin(), qsols[2 * ind + 1].end(), NAN);
    return;
  }
  const double cos_actmp = tmp;
  const double sin_actmp = sqrt(std::max(0.0, 1.0 - cos_actmp * cos_actmp));
  array<double, 3> k_C_O = {-r6[0] / l, -r6[1] / l, -r6[2] / l};
  array<double, 3> i_C_O = Cross(k_C_O, s6);
  tmp = Norm(i_C_O);
  i_C_O = {i_C_O[0] / tmp, i_C_O[1] / tmp, i_C_O[2] / tmp};
  array<double, 3> j_C_O = Cross(k_C_O, i_C_O);
  double ry = s6[0] * j_C_O[0] + s6[1] * j_C_O[1] + s6[2] * j_C_O[2];
  double rz = s6[0] * k_C_O[0] + s6[1] * k_C_O[1] + s6[2] * k_C_O[2];
  const double sa2 = sin_beta2 * cos_actmp + cos_beta2 * sin_actmp;
  const double ca2 = cos_beta2 * cos_actmp - sin_beta2 * sin_actmp;
  tmp = -rz * ca2 / (ry * sa2);
  if (!clamp_trig_roundoff(tmp)) {
    fill(qsols[2 * ind].begin(), qsols[2 * ind].end(), NAN);
    fill(qsols[2 * ind + 1].begin(), qsols[2 * ind + 1].end(), NAN);
    return;
  }
  const double sin_gamma = tmp;
  const double cos_gamma = sqrt(std::max(0.0, 1.0 - sin_gamma * sin_gamma));
  double v[3] = {-sa2 * cos_gamma, -sa2 * sin_gamma, -ca2};
  array<double, 3> s5;
  s5 = {i_C_O[0] * v[0] + j_C_O[0] * v[1] + k_C_O[0] * v[2],
        i_C_O[1] * v[0] + j_C_O[1] * v[1] + k_C_O[1] * v[2],
        i_C_O[2] * v[0] + j_C_O[2] * v[1] + k_C_O[2] * v[2]};
  if (branch == 1) {
    tmp = 2 * sa2 * cos_gamma;
    s5 = {s5[0] + tmp * i_C_O[0], s5[1] + tmp * i_C_O[1],
          s5[2] + tmp * i_C_O[2]};
  }
  array<double, 3> s4, r4, s3, s2;
  array<double, 6> sol1;
  array<double, 3> sol2;
  s4 = Cross(s5, r6);
  const double inverse_s4_norm = 1.0 / (l * fabs(sa2));
  s4 = {s4[0] * inverse_s4_norm, s4[1] * inverse_s4_norm,
        s4[2] * inverse_s4_norm};
  r4 = Cross(s5, s4);
  r4 = {r6[0] - d5 * s5[0] + a5 * r4[0], r6[1] - d5 * s5[1] + a5 * r4[1],
        r6[2] - d5 * s5[2] + a5 * r4[2]};
  rotate_by_beta1_scaled(s4, r4, s3);
  tmp = s3[1] * s3[1] + s3[0] * s3[0];
  if (tmp > SHOULDER_SING_TOL * SHOULDER_SING_TOL)
    s2 = shoulder_axis_from_s3(s3, tmp);
  else
    s2 = {sin(q1_sing), cos(q1_sing), 0};
  sol1 = q_from_axes(s2, s3, s4, s5, s6, k_E_O);
  sol2 = q_from_flipped_shoulder(sol1);
  qsols[2 * ind] = {sol1[0], sol1[1], sol1[2], sol1[3], sol1[4], sol1[5], q7};
  check_limits(qsols[2 * ind], 7, tuning);
  qsols[2 * ind + 1] = {sol2[0],           sol2[1],           sol2[2],
                        qsols[2 * ind][3], qsols[2 * ind][4], qsols[2 * ind][5],
                        qsols[2 * ind][6]};
  check_limits(qsols[2 * ind + 1], 3, tuning);
}

static double interpolate_swivel_q7(
    const unsigned int minimum, const unsigned int branch,
    const unsigned int n_points,
    const array<array<double, 2>, MAX_N_POINTS>& signed_errors,
    const array<array<double, 2>, MAX_N_POINTS>& q7s) {
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
    } else if (fabs(signed_errors[minimum + 1][branch]) <
               fabs(signed_errors[minimum - 1][branch])) {
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
        ((signed1 <= 0.0 && signed2 >= 0.0) ||
         (signed1 >= 0.0 && signed2 <= 0.0)) &&
        signed1 != signed2) {
      const double candidate = x1 - signed1 * (x2 - x1) / (signed2 - signed1);
      if (candidate >= x1 && candidate <= x2) return candidate;
    }
  }

  // If the signed coordinate is discontinuous here, retain GeoFIK's original
  // minimum-of-|error| refinement, generalized to the actual (possibly moved)
  // near-singularity sample positions rather than an assumed uniform grid.
  if (minimum > 1 && minimum < n_points - 2 &&
      fabs(signed_errors[minimum - 2][branch]) < ERR_THRESH &&
      fabs(signed_errors[minimum + 2][branch]) < ERR_THRESH) {
    const unsigned int first = fabs(signed_errors[minimum + 1][branch]) <
                                       fabs(signed_errors[minimum - 1][branch])
                                   ? minimum - 1
                                   : minimum - 2;
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
        const double candidate =
            (e2 - e1 + left_slope * x1 - right_slope * x2) / slope_difference;
        if (candidate > x1 && candidate < x2) q7_opt = candidate;
      }
    }
  }
  return q7_opt;
}

NANOGEOFIK_TARGET_CLONES
unsigned int franka_ik_swivel(const array<double, 3>& r,
                              const array<double, 9>& ROE, const double theta,
                              array<array<double, 7>, 8>& qsols,
                              const SolverTuning& tuning) {
  const double q1_sing = tuning.q1_sing;
  const unsigned int n_points = tuning.n_points;
  const unsigned int n_fine_search = tuning.n_fine_search;
  // IK with swivel angle as free variable (numerical)
  // INPUT: r = r_EO_O, position of frame E in frame O
  //        ROE, orientation of frame E in frame O (row-first format)
  //        theta, swivel angle (see paper for geometric defninition)
  //        qsols, array to store 8 solutions
  //        q1_sing, emergency value of q1 in case of singularity at shoulder
  //        joints (type-1 singularity). n_points, number of points to
  //        discretize the range of q7.
  // OUTPUT: number of solutions found.
  // NOTATION:
  // ri = r_iS_O,
  // si - s_i_O
  if (n_points < 2 || n_points > MAX_N_POINTS) {
    for (auto& solution : qsols) fill(solution.begin(), solution.end(), NAN);
    return 0;
  }
  array<double, 3> k_E_O = {ROE[2], ROE[5], ROE[8]};
  array<double, 3> r_O7S_O = {r[0] - dE * k_E_O[0], r[1] - dE * k_E_O[1],
                              r[2] - d1 - dE * k_E_O[2]};
  double tmp = Norm(r_O7S_O);
  if (tmp > b1 + b2 + a7) {
    for (int i = 0; i < 8; i++) fill(qsols[i].begin(), qsols[i].end(), NAN);
    return 0;
  }
  if (tmp < SING_TOL) {
    nanogeofik_log() << "ERROR: r_O7S_O is near zero";
    for (int i = 0; i < 8; i++) fill(qsols[i].begin(), qsols[i].end(), NAN);
    return 0;
  }
  array<double, 3> u_O7S_O = {r_O7S_O[0] / tmp, r_O7S_O[1] / tmp,
                              r_O7S_O[2] / tmp};
  array<double, 3> n1_O = stereographic_n1(u_O7S_O);
  const array<double, 3> sweep_i_E_O = {ROE[0], ROE[3], ROE[6]};
  const array<double, 3> sweep_j_E_O = {ROE[1], ROE[4], ROE[7]};
  array<double, 3> i_6_O;
  const SwivelGeometry geometry = build_swivel_geometry(
      sweep_i_E_O, sweep_j_E_O, n1_O, u_O7S_O, r_O7S_O, theta);
  SwivelSweep sweep;
  run_swivel_sweep(geometry, n_points, n_fine_search, tuning, sweep);

  const SwivelMinima& minima = sweep.minima;
  if (minima.count == 0) {
    for (int i = 0; i < 8; i++) fill(qsols[i].begin(), qsols[i].end(), NAN);
    return 0;
  }

  array<unsigned int, 2> m;
  for (unsigned int i = 0; i < minima.count; i++) {
    m = minima.values[i];
    const double q7_opt = interpolate_swivel_q7(m[0], m[1], n_points,
                                                sweep.signed_errors, sweep.q7s);
    franka_ik_q7_one_sol(q7_opt, sweep_i_E_O, sweep_j_E_O, k_E_O, i_6_O,
                         r_O7S_O, m[1], qsols, i, tuning);
  }
  for (int i = 2 * minima.count; i < 8; ++i) {
    fill(qsols[i].begin(), qsols[i].end(), NAN);
  }
  return 2 * minima.count;
}

double franka_swivel(const array<double, 7>& q) {
  // swivel angle for a configuration q using stereographic SEW parameterization
  array<Eigen::Matrix4d, 9> Ts;
  franka_fk_all_frames(Ts, q);
  array<double, 3> r4 = {Ts[3](0, 3), Ts[3](1, 3), Ts[3](2, 3) - d1};
  array<double, 3> r7 = {Ts[6](0, 3), Ts[6](1, 3), Ts[6](2, 3) - d1};
  array<double, 3> s4 = {Ts[3](0, 2), Ts[3](1, 2), Ts[3](2, 2)};
  double tmp = Norm(r7);
  if (tmp < SING_TOL) {
    nanogeofik_log() << "ERROR: r7 is near zero";
    return NAN;
  }
  array<double, 3> u_O7S_O = {r7[0] / tmp, r7[1] / tmp, r7[2] / tmp};
  array<double, 3> n1_O = stereographic_n1(u_O7S_O);
  array<double, 3> n2_O;
  Cross_(r7, r4, n2_O);
  tmp = Dot(n2_O, s4);
  if (tmp < 0) n2_O = {-n2_O[0], -n2_O[1], -n2_O[2]};
  return signed_angle(n1_O, n2_O, u_O7S_O);
}

// FUNCTIONS FOR JACOBIAN MATRIX
// ==========================================================================

NANOGEOFIK_TARGET_CLONES
unsigned int franka_J_ik_q7(const array<double, 3>& r,
                            const array<double, 9>& ROE, const double q7,
                            array<array<array<double, 6>, 7>, 8>& Jsols,
                            array<array<double, 7>, 8>& qsols,
                            const Output output, const Frame Jacobian_ee,
                            const SolverTuning& tuning) {
  const bool joint_angles = (output == Output::JointsAndJacobian);
  const double q1_sing = tuning.q1_sing;
  // IK to calculate Jacobian and joint angles with q7 as free variable.
  // INPUT: r = r_EO_O, position of frame E in frame O
  //        ROE, orientation of frame E in frame O (row-first format)
  //        q7, value of joint angle of joint 7
  //        Jsols, array to store 8 Jacobian solutions
  //        qsols, array to store 8 joint-angle solutions
  //        joint_angles, if false only Jacobians are returned
  //        Jacobian_ee, end-effector frame of the Jacobian, not the IK. Only
  //        'E', 'F', '8' and '6' are supported. q1_sing, emergency value of q1
  //        in case of singularity at shoulder joints (type-1 singularity).
  // OUTPUT: number of solutions found.
  // NOTATION:
  // ri = r_iS_O,
  // si - s_i_O,
  const array<double, 3> i_E_O = {ROE[0], ROE[3], ROE[6]};
  const array<double, 3> j_E_O = {ROE[1], ROE[4], ROE[7]};
  const array<double, 3> k_E_O = {ROE[2], ROE[5], ROE[8]};
  array<double, 3> i_6_O;
  array<double, 3> s6;
  wrist_axes_from_q7(i_E_O, j_E_O, q7, i_6_O, s6);
  array<double, 3> r6 = {r[0] - dE * k_E_O[0] - a7 * i_6_O[0],
                         r[1] - dE * k_E_O[1] - a7 * i_6_O[1],
                         r[2] - d1 - dE * k_E_O[2] - a7 * i_6_O[2]};
  double l = Norm(r6);
  double tmp = (b1 * b1 - l * l - b2 * b2) / (-2 * l * b2);
  if (!clamp_trig_roundoff(tmp)) {
    nanogeofik_log() << "ERROR: unable to assembly kinematic chain";
    for (int i = 0; i < 8; ++i) {
      fill(qsols[i].begin(), qsols[i].end(), NAN);
      for (auto& row : Jsols[i]) fill(row.begin(), row.end(), NAN);
    }
    return 0;
  }
  const double cos_actmp = tmp;
  const double sin_actmp = sqrt(std::max(0.0, 1.0 - cos_actmp * cos_actmp));
  array<double, 3> k_C_O = {-r6[0] / l, -r6[1] / l, -r6[2] / l};
  array<double, 3> i_C_O = Cross(k_C_O, s6);

  tmp = Norm(i_C_O);
  i_C_O = {i_C_O[0] / tmp, i_C_O[1] / tmp, i_C_O[2] / tmp};
  array<double, 3> j_C_O = Cross(k_C_O, i_C_O);
  double ry = s6[0] * j_C_O[0] + s6[1] * j_C_O[1] + s6[2] * j_C_O[2];
  double rz = s6[0] * k_C_O[0] + s6[1] * k_C_O[1] + s6[2] * k_C_O[2];
  array<array<double, 3>, 4> s5s;
  array<double, 4> inverse_s4_norms;
  double sa2, ca2;
  int n_alphs = 1;
  unsigned int n_sols = 0;
  if (d3 + d5 < l && l < b1 + b2) n_alphs = 2;
  double v[3];
  for (int i = 0; i < n_alphs; i++) {
    // Same `continue`-skips-the-advance shape as in
    // franka_ik_q7. See README.md.
    const double branch_sin_actmp = i == 0 ? sin_actmp : -sin_actmp;
    sa2 = sin_beta2 * cos_actmp + cos_beta2 * branch_sin_actmp;
    ca2 = cos_beta2 * cos_actmp - sin_beta2 * branch_sin_actmp;
    tmp = -rz * ca2 / (ry * sa2);
    if (!clamp_trig_roundoff(tmp)) continue;
    const double sin_gamma = tmp;
    const double cos_gamma = sqrt(std::max(0.0, 1.0 - sin_gamma * sin_gamma));
    v[0] = -sa2 * cos_gamma;
    v[1] = -sa2 * sin_gamma;
    v[2] = -ca2;
    s5s[n_sols] = {i_C_O[0] * v[0] + j_C_O[0] * v[1] + k_C_O[0] * v[2],
                   i_C_O[1] * v[0] + j_C_O[1] * v[1] + k_C_O[1] * v[2],
                   i_C_O[2] * v[0] + j_C_O[2] * v[1] + k_C_O[2] * v[2]};
    tmp = 2 * sa2 * cos_gamma;
    // s5[n_sols+1] = s5s[n_sols] + (2*sa2*cos(tmp)*i_C_O);
    s5s[n_sols + 1] = {s5s[n_sols][0] + tmp * i_C_O[0],
                       s5s[n_sols][1] + tmp * i_C_O[1],
                       s5s[n_sols][2] + tmp * i_C_O[2]};
    inverse_s4_norms[n_sols] = inverse_s4_norms[n_sols + 1] =
        1.0 / (l * fabs(sa2));
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
    s4 = {s4[0] * inverse_s4_norms[i], s4[1] * inverse_s4_norms[i],
          s4[2] * inverse_s4_norms[i]};
    // r4 = r6 - d5 * s5 + a5 * Cross(s5, s4);
    Cross_(s5, s4, r4);
    r4 = {r6[0] - d5 * s5[0] + a5 * r4[0], r6[1] - d5 * s5[1] + a5 * r4[1],
          r6[2] - d5 * s5[2] + a5 * r4[2]};
    rotate_by_beta1_scaled(s4, r4, s3);
    tmp = s3[1] * s3[1] + s3[0] * s3[0];
    if (tmp > SHOULDER_SING_TOL * SHOULDER_SING_TOL) {
      s2 = shoulder_axis_from_s3(s3, tmp);
    } else {
      s2 = {sin(q1_sing), cos(q1_sing), 0};
    }
    save_J_sol(s2, s3, s4, s5, s6, k_E_O, r4, r6, r, Jsols, i, Jacobian_ee);
    if (joint_angles) {
      sol1 = q_from_axes(s2, s3, s4, s5, s6, k_E_O);
      sol2 = q_from_flipped_shoulder(sol1);
      qsols[2 * i] = {sol1[0], sol1[1], sol1[2], sol1[3], sol1[4], sol1[5], q7};
      check_limits(qsols[2 * i], 7, tuning);
      qsols[2 * i + 1] = {sol2[0],         sol2[1],         sol2[2],
                          qsols[2 * i][3], qsols[2 * i][4], qsols[2 * i][5],
                          qsols[2 * i][6]};
      check_limits(qsols[2 * i + 1], 3, tuning);
    }
  }
  for (int i = 2 * n_sols; i < 8; ++i) {
    for (auto& row : Jsols[i]) fill(row.begin(), row.end(), NAN);
  }
  for (int i = joint_angles ? 2 * n_sols : 0; i < 8; i++)
    fill(qsols[i].begin(), qsols[i].end(), NAN);
  return 2 * n_sols;
}

NANOGEOFIK_TARGET_CLONES
unsigned int franka_J_ik_q4(const array<double, 3>& r,
                            const array<double, 9>& ROE, const double q4,
                            array<array<array<double, 6>, 7>, 8>& Jsols,
                            array<array<double, 7>, 8>& qsols,
                            const Output output, const Frame Jacobian_ee,
                            const SolverTuning& tuning) {
  const bool joint_angles = (output == Output::JointsAndJacobian);
  const double q1_sing = tuning.q1_sing;
  const double q7_sing = tuning.q7_sing;
  // IK to calculate Jacobian and joint angles with q4 as free variable.
  // INPUT: r = r_EO_O, position of frame E in frame O
  //        ROE, orientation of frame E in frame O (row-first format)
  //        q4, value of joint angle of joint 4
  //        Jsols, array to store 8 Jacobian solutions
  //        qsols, array to store 8 joint-angle solutions
  //        joint_angles, if false only Jacobians are returned
  //        Jacobian_ee, end-effector frame of the Jacobian, not the IK. Only
  //        'E', 'F', '8' and '6' are supported. q1_sing, emergency value of q1
  //        in case of singularity at shoulder joints (type-1 singularity).
  //        q7_sing, emergency value of q7 in case S7 intersects S (type-2
  //        singularity)
  // OUTPUT: number of solutions found.
  // NOTATION:
  // ri = r_iS_O,
  // si - s_i_O,
  array<double, 3> r_ES_O = {r[0], r[1], r[2] - d1};
  array<double, 3> tmp_v = {r_ES_O[1] * ROE[8] - r_ES_O[2] * ROE[5],
                            r_ES_O[2] * ROE[2] - r_ES_O[0] * ROE[8],
                            r_ES_O[0] * ROE[5] - r_ES_O[1] * ROE[2]};
  if (tmp_v[0] * tmp_v[0] + tmp_v[1] * tmp_v[1] + tmp_v[2] * tmp_v[2] <
      SING_TOL * SING_TOL)
    return franka_J_ik_q7(r, ROE, q7_sing, Jsols, qsols, output, Jacobian_ee,
                          SolverTuning{q1_sing});
  array<double, 3> r_O7S_O = {r_ES_O[0] - dE * ROE[2], r_ES_O[1] - dE * ROE[5],
                              r_ES_O[2] - dE * ROE[8]};
  array<double, 3> r_O7S_E = {
      ROE[0] * r_O7S_O[0] + ROE[3] * r_O7S_O[1] + ROE[6] * r_O7S_O[2],
      ROE[1] * r_O7S_O[0] + ROE[4] * r_O7S_O[1] + ROE[7] * r_O7S_O[2],
      ROE[2] * r_O7S_O[0] + ROE[5] * r_O7S_O[1] + ROE[8] * r_O7S_O[2]};
  const double alpha = q4 + beta1 + beta2 - PI;
  const double sin_alpha = sin(alpha);
  const double cos_alpha = cos(alpha);
  double lo2 = b1 * b1 + b2 * b2 - 2 * b1 * b2 * cos_alpha;
  double lp2 = lo2 - r_O7S_E[2] * r_O7S_E[2];
  if (lp2 < 0) {
    if (lp2 * lp2 < SING_TOL * SING_TOL)
      lp2 = 0;
    else {
      nanogeofik_log() << "\nERROR: unable to assembly kinematic chain\n";
      for (int i = 0; i < 8; ++i) fill(qsols[i].begin(), qsols[i].end(), NAN);
      return 0;
    }
  }
  double sin_gamma_offset = b1 * sin_alpha / sqrt(lo2);
  if (!clamp_trig_roundoff(sin_gamma_offset)) {
    for (unsigned int i = 0; i < qsols.size(); ++i) {
      fill(qsols[i].begin(), qsols[i].end(), NAN);
      for (auto& row : Jsols[i]) fill(row.begin(), row.end(), NAN);
    }
    return 0;
  }
  const double cos_gamma_offset =
      sqrt(std::max(0.0, 1.0 - sin_gamma_offset * sin_gamma_offset));
  const double cg2 =
      cos_beta2 * cos_gamma_offset - sin_beta2 * sin_gamma_offset;
  const double sg2 =
      sin_beta2 * cos_gamma_offset + cos_beta2 * sin_gamma_offset;
  const double Lp2 = r_O7S_E[0] * r_O7S_E[0] + r_O7S_E[1] * r_O7S_E[1];
  const double sqrt_Lp2 = sqrt(Lp2);
  const double phi = nanogeofik_atan::atan2(-r_O7S_E[1], -r_O7S_E[0]);
  double tmp = (Lp2 + a7 * a7 - lp2) / (2 * sqrt_Lp2 * a7);
  if (tmp > 1.0) {
    if ((tmp - 1) * (tmp - 1) < SING_TOL * SING_TOL)
      tmp = 1.0;
    else {
      nanogeofik_log() << "\nERROR: unable to assembly kinematic chain\n";
      for (int i = 0; i < 8; ++i) {
        fill(qsols[i].begin(), qsols[i].end(), NAN);
      }
      return 0;
    }
  }
  const double psi = acos(tmp);
  const double sin_psi = sqrt(std::max(0.0, 1.0 - tmp * tmp));
  const double cos_phi = -r_O7S_E[0] / sqrt_Lp2;
  const double sin_phi = -r_O7S_E[1] / sqrt_Lp2;
  const double wrist_cosines[2] = {sin_phi * tmp + cos_phi * sin_psi,
                                   sin_phi * tmp - cos_phi * sin_psi};
  const double wrist_sines[2] = {-(cos_phi * tmp - sin_phi * sin_psi),
                                 -(cos_phi * tmp + sin_phi * sin_psi)};
  double ry, rz;
  double q7s[2] = {-phi - psi - 3 * PI / 4, -phi + psi - 3 * PI / 4};
  size_t ind = 0;
  array<double, 3> s2, s3, s4, s5, s6, r4, r6, i_C_O, j_C_O, k_C_O;
  array<double, 3> s7 = {ROE[2], ROE[5], ROE[8]};
  array<double, 6> sol1;
  array<double, 3> sol2;
  for (unsigned int q7_index = 0; q7_index < 2; ++q7_index) {
    const double q7 = q7s[q7_index];
    const double wrist_cos = wrist_cosines[q7_index];
    const double wrist_sin = wrist_sines[q7_index];
    tmp_v = {wrist_cos, wrist_sin, 0};
    s6 = {ROE[0] * tmp_v[0] + ROE[1] * tmp_v[1],
          ROE[3] * tmp_v[0] + ROE[4] * tmp_v[1],
          ROE[6] * tmp_v[0] + ROE[7] * tmp_v[1]};
    tmp_v = {-a7 * wrist_sin, a7 * wrist_cos, 0};
    r6 = {ROE[0] * tmp_v[0] + ROE[1] * tmp_v[1],
          ROE[3] * tmp_v[0] + ROE[4] * tmp_v[1],
          ROE[6] * tmp_v[0] + ROE[7] * tmp_v[1]};
    r6 = {r6[0] + r_O7S_O[0], r6[1] + r_O7S_O[1], r6[2] + r_O7S_O[2]};
    const double r6_norm = Norm(r6);
    k_C_O = {-r6[0] / r6_norm, -r6[1] / r6_norm, -r6[2] / r6_norm};
    Cross_(k_C_O, s6, i_C_O);
    tmp = Norm(i_C_O);
    i_C_O = {i_C_O[0] / tmp, i_C_O[1] / tmp, i_C_O[2] / tmp};
    Cross_(k_C_O, i_C_O, j_C_O);
    ry = s6[0] * j_C_O[0] + s6[1] * j_C_O[1] + s6[2] * j_C_O[2];
    rz = s6[0] * k_C_O[0] + s6[1] * k_C_O[1] + s6[2] * k_C_O[2];
    tmp = -rz * cg2 / (ry * sg2);
    if (tmp * tmp > 1) continue;
    const double sin_gamma = tmp;
    const double cos_gamma = sqrt(std::max(0.0, 1.0 - sin_gamma * sin_gamma));
    for (int gamma_branch = 0; gamma_branch < 2; ++gamma_branch) {
      tmp_v = {(gamma_branch == 0 ? -sg2 : sg2) * cos_gamma, -sg2 * sin_gamma,
               -cg2};
      s5 = {i_C_O[0] * tmp_v[0] + j_C_O[0] * tmp_v[1] + k_C_O[0] * tmp_v[2],
            i_C_O[1] * tmp_v[0] + j_C_O[1] * tmp_v[1] + k_C_O[1] * tmp_v[2],
            i_C_O[2] * tmp_v[0] + j_C_O[2] * tmp_v[1] + k_C_O[2] * tmp_v[2]};
      Cross_(s5, r6, s4);
      const double inverse_s4_norm = 1.0 / (r6_norm * fabs(sg2));
      s4 = {s4[0] * inverse_s4_norm, s4[1] * inverse_s4_norm,
            s4[2] * inverse_s4_norm};
      Cross_(s5, s4, r4);
      r4 = {r6[0] - d5 * s5[0] + a5 * r4[0], r6[1] - d5 * s5[1] + a5 * r4[1],
            r6[2] - d5 * s5[2] + a5 * r4[2]};
      rotate_by_beta1_scaled(s4, r4, s3);
      tmp = s3[1] * s3[1] + s3[0] * s3[0];
      if (tmp > SHOULDER_SING_TOL * SHOULDER_SING_TOL)
        s2 = shoulder_axis_from_s3(s3, tmp);
      else
        s2 = {sin(q1_sing), cos(q1_sing), 0};
      save_J_sol(s2, s3, s4, s5, s6, s7, r4, r6, r, Jsols, ind, Jacobian_ee);
      if (joint_angles) {
        sol1 = q_from_axes_with_q4(s2, s3, s4, s5, s6, s7, q4);
        sol2 = q_from_flipped_shoulder(sol1);
        qsols[2 * ind] = {sol1[0], sol1[1], sol1[2], sol1[3],
                          sol1[4], sol1[5], q7};
        check_limits(qsols[2 * ind], 7, tuning);
        qsols[2 * ind + 1] = {sol2[0],           sol2[1],
                              sol2[2],           qsols[2 * ind][3],
                              qsols[2 * ind][4], qsols[2 * ind][5],
                              qsols[2 * ind][6]};
        check_limits(qsols[2 * ind + 1], 3, tuning);
      }
      ind++;
    }
  }
  for (int i = 2 * ind; i < 8; ++i) {
    for (auto& row : Jsols[i]) fill(row.begin(), row.end(), NAN);
  }
  for (int i = joint_angles ? 2 * ind : 0; i < 8; i++)
    fill(qsols[i].begin(), qsols[i].end(), NAN);
  return 2 * ind;
}

static unsigned int franka_J_ik_q6_parallel(
    const array<double, 3>& r, const array<double, 3>& r_ES_O,
    const array<double, 9>& ROE, const int sgn,
    array<array<array<double, 6>, 7>, 8>& Jsols,
    array<array<double, 7>, 8>& qsols, const bool joint_angles,
    const Frame Jacobian_ee, const SolverTuning& tuning) {
  const double q1_sing = tuning.q1_sing;
  // Parallel case of the Jacobian IK with q6 as free variable. Only called by
  // franka_J_ik_q6(), not by the user. INPUT: r, r_ES_O, ROE, sgn  =
  // sign(cos(q6)), Jsols, qsols, joint_angles, Jacobian_ee, q1_sing. OUTPUT:
  // number of solutions found. NOTATION: ri = r_iS_O, i = 1,2,3,4,5,6,7 si =
  // s_i_O Q is a frame that is parallel to frame E and has origin at Q (Q is
  // called E' in the paper)
  array<double, 3> s7 = {ROE[2], ROE[5], ROE[8]};
  array<double, 3> r_QS_O = {r_ES_O[0] + (-dE + sgn * d5) * s7[0],
                             r_ES_O[1] + (-dE + sgn * d5) * s7[1],
                             r_ES_O[2] + (-dE + sgn * d5) * s7[2]};
  array<double, 3> r_SQ_Q = {
      -ROE[0] * r_QS_O[0] - ROE[3] * r_QS_O[1] - ROE[6] * r_QS_O[2],
      -ROE[1] * r_QS_O[0] - ROE[4] * r_QS_O[1] - ROE[7] * r_QS_O[2],
      -ROE[2] * r_QS_O[0] - ROE[5] * r_QS_O[1] - ROE[8] * r_QS_O[2]};
  double tmp = b1 * b1 - r_SQ_Q[2] * r_SQ_Q[2];
  if (tmp * tmp < SING_TOL * SING_TOL) tmp = 0;
  if (tmp < 0) {
    nanogeofik_log() << "ERROR: unable to assembly kinematic chain";
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
  array<double, 3> tmp_v, r_O6pQ_Q, i_4_Q, r_O4Q_Q, s6_Q, r_O6Q_Q, s4_Q, s3_Q,
      s2, s3, s4, s5, s6, r4, r6;
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
      r_O6pQ_Q = {a7 * r_O6pQ_Q[0] / l_SpQ, a7 * r_O6pQ_Q[1] / l_SpQ,
                  a7 * r_O6pQ_Q[2] / l_SpQ};
      i_4_Q = {r_SpQ_Q[0] - r_O6pQ_Q[0], r_SpQ_Q[1] - r_O6pQ_Q[1],
               r_SpQ_Q[2] - r_O6pQ_Q[2]};
      tmp = Norm(i_4_Q);
      tmp_sgn = L < 0 ? -1 : 1;
      i_4_Q = {tmp_sgn * i_4_Q[0] / tmp, tmp_sgn * i_4_Q[1] / tmp,
               tmp_sgn * i_4_Q[2] / tmp};
      r_O4Q_Q = {r_O6pQ_Q[0] + a5 * i_4_Q[0], r_O6pQ_Q[1] + a5 * i_4_Q[1],
                 r_O6pQ_Q[2] + a5 * i_4_Q[2]};
      Cross_(r_O6pQ_Q, k, s6_Q);
      s6_Q = {s6_Q[0] / a7, s6_Q[1] / a7, s6_Q[2] / a7};
      r_O6Q_Q = {r_O6pQ_Q[0], r_O6pQ_Q[1], r_O6pQ_Q[2] - sgn * d5};
      rs << r_O4Q_Q[0], r_O6Q_Q[0], r_O4Q_Q[1], r_O6Q_Q[1], r_O4Q_Q[2],
          r_O6Q_Q[2];
      rs = ROQ * rs;  // r_O4Q_O, r_O6Q_O
      r4 = {rs(0, 0) + r_QS_O[0], rs(1, 0) + r_QS_O[1], rs(2, 0) + r_QS_O[2]};
      r6 = {rs(0, 1) + r_QS_O[0], rs(1, 1) + r_QS_O[1], rs(2, 1) + r_QS_O[2]};
      Cross_(i_4_Q, s5_Q, s4_Q);
      tmp_v = {r_O4Q_Q[0] - r_SQ_Q[0], r_O4Q_Q[1] - r_SQ_Q[1],
               r_O4Q_Q[2] - r_SQ_Q[2]};
      rotate_by_sin_cos(s4_Q, sin_beta1, cos_beta1, tmp_v, s3_Q);
      tmp = Norm(s3_Q);
      // s3_Q = {s3_Q[0]/tmp,s3_Q[1]/tmp,s3_Q[2]/tmp};
      partial_J_Q << s3_Q[0] / tmp, s4_Q[0], s5_Q[0], s6_Q[0], s3_Q[1] / tmp,
          s4_Q[1], s5_Q[1], s6_Q[1], s3_Q[2] / tmp, s4_Q[2], s5_Q[2], s6_Q[2];
      partial_J_O = ROQ * partial_J_Q;
      s3 = {partial_J_O(0, 0), partial_J_O(1, 0), partial_J_O(2, 0)};
      s4 = {partial_J_O(0, 1), partial_J_O(1, 1), partial_J_O(2, 1)};
      s5 = {partial_J_O(0, 2), partial_J_O(1, 2), partial_J_O(2, 2)};
      s6 = {partial_J_O(0, 3), partial_J_O(1, 3), partial_J_O(2, 3)};
      q7 = nanogeofik_atan::atan2(r_O6pQ_Q[1], -r_O6pQ_Q[0]) + PI / 4;
      tmp = s3[1] * s3[1] + s3[0] * s3[0];
      if (tmp > SHOULDER_SING_TOL * SHOULDER_SING_TOL)
        s2 = shoulder_axis_from_s3(s3, tmp);
      else
        s2 = {sin(q1_sing), cos(q1_sing), 0};
      save_J_sol(s2, s3, s4, s5, s6, s7, r4, r6, r, Jsols, ind, Jacobian_ee);
      if (joint_angles) {
        sol1 = q_from_axes(s2, s3, s4, s5, s6, s7);
        sol2 = q_from_flipped_shoulder(sol1);
        qsols[2 * ind] = {sol1[0], sol1[1], sol1[2], sol1[3],
                          sol1[4], sol1[5], q7};
        check_limits(qsols[2 * ind], 7, tuning);
        qsols[2 * ind + 1] = {sol2[0],           sol2[1],
                              sol2[2],           qsols[2 * ind][3],
                              qsols[2 * ind][4], qsols[2 * ind][5],
                              qsols[2 * ind][6]};
        check_limits(qsols[2 * ind + 1], 3, tuning);
      }
      ind++;
    }
  }
  for (int i = 2 * ind; i < 8; ++i) {
    for (auto& row : Jsols[i]) fill(row.begin(), row.end(), NAN);
  }
  for (int i = joint_angles ? 2 * ind : 0; i < 8; i++)
    fill(qsols[i].begin(), qsols[i].end(), NAN);
  return 2 * ind;
}

NANOGEOFIK_TARGET_CLONES
unsigned int franka_J_ik_q6(const array<double, 3>& r,
                            const array<double, 9>& ROE, const double q6,
                            array<array<array<double, 6>, 7>, 8>& Jsols,
                            array<array<double, 7>, 8>& qsols,
                            const Output output, const Frame Jacobian_ee,
                            const SolverTuning& tuning) {
  const bool joint_angles = (output == Output::JointsAndJacobian);
  const double q1_sing = tuning.q1_sing;
  const double q7_sing = tuning.q7_sing;
  // IK to calculate Jacobian and joint angles with q6 as free variable.
  // INPUT: r = r_EO_O, position of frame E in frame O
  //        ROE, orientation of frame E in frame O (row-first format)
  //        q6, value of joint angle of joint 6
  //        Jsols, array to store 8 Jacobian solutions
  //        qsols, array to store 8 joint-angle solutions
  //        joint_angles, if false only Jacobians are returned
  //        Jacobian_ee, end-effector frame of the Jacobian, not the IK. Only
  //        'E', 'F', '8' and '6' are supported. q1_sing, emergency value of q1
  //        in case of singularity at shoulder joints (type-1 singularity).
  //        q7_sing, emergency value of q7 in case S7 intersects S (type-2
  //        singularity)
  // OUTPUT: number of solutions found.
  // NOTATION:
  // ri = r_iS_O,
  // si - s_i_O,
  array<double, 3> r_ES_O = {r[0], r[1], r[2] - d1};
  array<double, 3> tmp_v = {r_ES_O[1] * ROE[8] - r_ES_O[2] * ROE[5],
                            r_ES_O[2] * ROE[2] - r_ES_O[0] * ROE[8],
                            r_ES_O[0] * ROE[5] - r_ES_O[1] * ROE[2]};
  if (tmp_v[0] * tmp_v[0] + tmp_v[1] * tmp_v[1] + tmp_v[2] * tmp_v[2] <
      SING_TOL * SING_TOL)
    return franka_J_ik_q7(r, ROE, q7_sing, Jsols, qsols, output, Jacobian_ee,
                          tuning);
  const double sg1 = sin(q6);  // sin(pi-q6)
  const double cos_q6 = cos(q6);
  // Parallel branch first in the near-parallel sliver (see franka_ik_q6):
  // it is exact there, and only falls through to the closed form when it
  // cannot assemble.
  if (sg1 * sg1 < WRIST_PARALLEL_TOL * WRIST_PARALLEL_TOL) {
    const unsigned int n_parallel =
        franka_J_ik_q6_parallel(r, r_ES_O, ROE, cos_q6 >= 0 ? 1 : -1, Jsols,
                                qsols, joint_angles, Jacobian_ee, tuning);
    if (n_parallel > 0) return n_parallel;
  }
  // NON-PARALLEL CASE:
  array<double, 3> s7 = {ROE[2], ROE[5], ROE[8]};
  const double cg1 = -cos_q6;  // cos(pi-q6)
  array<double, 3> r_O7S_O = {r_ES_O[0] - dE * ROE[2], r_ES_O[1] - dE * ROE[5],
                              r_ES_O[2] - dE * ROE[8]};
  array<double, 3> r_PS_O = {r_O7S_O[0] + (a7 * cg1 / sg1) * s7[0],
                             r_O7S_O[1] + (a7 * cg1 / sg1) * s7[1],
                             r_O7S_O[2] + (a7 * cg1 / sg1) * s7[2]};
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
      nanogeofik_log() << "ERROR: unable to assembly kinematic chain";
      for (int i = 0; i < 8; ++i) {
        fill(qsols[i].begin(), qsols[i].end(), NAN);
        for (auto& row : Jsols[i]) fill(row.begin(), row.end(), NAN);
      }
      return 0;
    }
  }
  // Same Heron-stable triangle and gamma2 composition as franka_ik_q6:
  // every factor is a difference of like-sized lengths, so precision
  // survives the near-degenerate triangles of the wrist-near-parallel band.
  const double lP_minus_c = lP - c;
  const double lP_plus_c = lP + c;
  const double sin_tau =
      sqrt(std::max(0.0, (b1 - lP_minus_c) * (b1 + lP_minus_c) *
                             (lP_plus_c - b1) * (lP_plus_c + b1))) /
      (2 * lP * c);
  unsigned int n_gamma_sols = 1;
  if ((d3 + d5 + lC < lP) && (lP < b1 + c)) n_gamma_sols = 2;
  const double base_numerator = d5 + lC;
  double cos_gamma2s[2] = {(base_numerator * tmp - a5 * sin_tau) / c,
                           (base_numerator * tmp + a5 * sin_tau) / c};
  double sin_gamma2s[2] = {(sin_tau * base_numerator + tmp * a5) / c,
                           (sin_tau * base_numerator - tmp * a5) / c};
  const double cone_denominator = fabs(sg1) * sqrt(Cx * Cx + Cy * Cy);
  const double u2_x = Cx * sg1;
  const double u2_y = Cy * sg1;
  const double u2 = nanogeofik_atan::atan2(u2_x, u2_y);
  const double sin_u2 = u2_x / cone_denominator;
  const double cos_u2 = u2_y / cone_denominator;
  array<array<double, 3>, 4> s5s;
  array<double, 4> inverse_s4_norms;
  double q7s[4];
  double d, u1;
  unsigned int n_sols = 0;
  for (unsigned int i = 0; i < n_gamma_sols; i++) {
    d = lP * cos_gamma2s[i];
    tmp = (d + Cz * cg1) / cone_denominator;
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
    const double sin_u1 = tmp;
    const double cos_u1 = sqrt(std::max(0.0, 1.0 - sin_u1 * sin_u1));
    q7s[n_sols] = 5 * PI / 4 - u1 + u2;
    tmp_v = {-sg1 * (cos_u1 * cos_u2 + sin_u1 * sin_u2),
             -sg1 * (sin_u1 * cos_u2 - cos_u1 * sin_u2), cg1};
    column_1s_times_vec(ROE, tmp_v, s5s[n_sols]);
    const double inverse_s4_norm = 1.0 / (lP * fabs(sin_gamma2s[i]));
    inverse_s4_norms[n_sols] = inverse_s4_norm;
    n_sols++;
    q7s[n_sols] = PI / 4 + u1 + u2;
    tmp_v = {sg1 * (cos_u1 * cos_u2 - sin_u1 * sin_u2),
             -sg1 * (sin_u1 * cos_u2 + cos_u1 * sin_u2), cg1};
    column_1s_times_vec(ROE, tmp_v, s5s[n_sols]);
    inverse_s4_norms[n_sols] = inverse_s4_norm;
    n_sols++;
  }
  array<double, 3> s2, s3, s4, s6, r4, r6;
  array<double, 6> sol1;
  array<double, 3> sol2;
  unsigned int assembled_sols = 0;
  for (unsigned int i = 0; i < n_sols; i++) {
    r6 = {r_PS_O[0] - lC * s5s[i][0], r_PS_O[1] - lC * s5s[i][1],
          r_PS_O[2] - lC * s5s[i][2]};
    tmp_v = {r_O7S_O[0] - r6[0], r_O7S_O[1] - r6[1], r_O7S_O[2] - r6[2]};
    Cross_(s7, tmp_v, s6);
    s6 = {s6[0] * inv_a7, s6[1] * inv_a7, s6[2] * inv_a7};
    Cross_(s5s[i], r6, s4);
    s4 = {s4[0] * inverse_s4_norms[i], s4[1] * inverse_s4_norms[i],
          s4[2] * inverse_s4_norms[i]};
    Cross_(s5s[i], s4, tmp_v);
    r4 = {r6[0] - d5 * s5s[i][0] + a5 * tmp_v[0],
          r6[1] - d5 * s5s[i][1] + a5 * tmp_v[1],
          r6[2] - d5 * s5s[i][2] + a5 * tmp_v[2]};
    tmp = Norm(r4);
    // See the joint-angle solver above: reject algebraic roots that cannot
    // assemble the fixed-length upper-arm link.
    if (fabs(tmp - b1) > SING_TOL) continue;
    rotate_by_sin_cos(s4, sin_beta1, cos_beta1, r4, s3);
    s3 = {s3[0] / tmp, s3[1] / tmp, s3[2] / tmp};
    tmp = s3[1] * s3[1] + s3[0] * s3[0];
    if (tmp > SHOULDER_SING_TOL * SHOULDER_SING_TOL)
      s2 = shoulder_axis_from_s3(s3, tmp);
    else
      s2 = {sin(q1_sing), cos(q1_sing), 0};
    save_J_sol(s2, s3, s4, s5s[i], s6, s7, r4, r6, r, Jsols, assembled_sols,
               Jacobian_ee);
    if (joint_angles) {
      sol1 = q_from_axes_with_q6(s2, s3, s4, s5s[i], s6, q6);
      sol2 = q_from_flipped_shoulder(sol1);
      qsols[2 * assembled_sols] = {sol1[0], sol1[1], sol1[2], sol1[3],
                                   sol1[4], sol1[5], q7s[i]};
      check_limits(qsols[2 * assembled_sols], 7, tuning);
      qsols[2 * assembled_sols + 1] = {sol2[0],
                                       sol2[1],
                                       sol2[2],
                                       qsols[2 * assembled_sols][3],
                                       qsols[2 * assembled_sols][4],
                                       qsols[2 * assembled_sols][5],
                                       qsols[2 * assembled_sols][6]};
      check_limits(qsols[2 * assembled_sols + 1], 3, tuning);
    }
    assembled_sols++;
  }
  for (int i = 2 * assembled_sols; i < 8; ++i) {
    for (auto& row : Jsols[i]) fill(row.begin(), row.end(), NAN);
  }
  for (int i = joint_angles ? 2 * assembled_sols : 0; i < 8; i++)
    fill(qsols[i].begin(), qsols[i].end(), NAN);
  return 2 * assembled_sols;
}

NANOGEOFIK_TARGET_CLONES
unsigned int franka_J_ik_q5(const array<double, 3>& r,
                            const array<double, 9>& ROE, const double q5,
                            array<array<array<double, 6>, 7>, 8>& Jsols,
                            array<array<double, 7>, 8>& qsols,
                            const Output output, const Frame Jacobian_ee,
                            const SolverTuning& tuning) {
  const bool joint_angles = (output == Output::JointsAndJacobian);
  const double q1_sing = tuning.q1_sing;
  const double q7_sing = tuning.q7_sing;
  // Closed-form analytical IK with q5 as free variable via quartic reduction.
  // Same geometry and conventions as franka_ik_q5(); Jacobians are saved as
  // a byproduct of the screw axes, like in the other J solvers.
  for (auto& J : Jsols)
    for (auto& row : J) fill(row.begin(), row.end(), NAN);
  for (auto& solution : qsols) fill(solution.begin(), solution.end(), NAN);
  const array<double, 3> k_E_O = {ROE[2], ROE[5], ROE[8]};
  const array<double, 3> r_ES_O = {r[0], r[1], r[2] - d1};
  // Type-2 singularity: delegate to the q7 solver (see franka_ik_q5()).
  array<double, 3> tmp_v = {r_ES_O[1] * k_E_O[2] - r_ES_O[2] * k_E_O[1],
                            r_ES_O[2] * k_E_O[0] - r_ES_O[0] * k_E_O[2],
                            r_ES_O[0] * k_E_O[1] - r_ES_O[1] * k_E_O[0]};
  if (tmp_v[0] * tmp_v[0] + tmp_v[1] * tmp_v[1] + tmp_v[2] * tmp_v[2] <
      SING_TOL * SING_TOL)
    return franka_J_ik_q7(r, ROE, q7_sing, Jsols, qsols, output, Jacobian_ee,
                          SolverTuning{q1_sing});
  const array<double, 3> i_E_O = {ROE[0], ROE[3], ROE[6]};
  const array<double, 3> j_E_O = {ROE[1], ROE[4], ROE[7]};
  const array<double, 3> r_O7S_O = {r_ES_O[0] - dE * k_E_O[0],
                                    r_ES_O[1] - dE * k_E_O[1],
                                    r_ES_O[2] - dE * k_E_O[2]};
  Q5Root roots[Q5_MAX_ROOTS];
  double sin_q5, cos_q5;
  nanogeofik_sincos::sincos(q5, sin_q5, cos_q5);
  const unsigned int n_roots =
      find_q5_roots(i_E_O, j_E_O, r_O7S_O, sin_q5, roots, tuning);
  array<double, 3> s2, s3, s4, s5, s6, r4, r6;
  unsigned int assembled_sols = 0;
  // Each root fills two of the eight solution rows; see franka_ik_q5().
  for (unsigned int i = 0; i < n_roots && assembled_sols + 2 <= 8; ++i) {
    if (!assemble_q5_arm(roots[i], i_E_O, j_E_O, r_O7S_O, cos_q5, q1_sing, s2,
                         s3, s4, s5, s6, r4, r6))
      continue;
    if (fabs(q5_wrap_two_pi(signed_angle(s4, s6, s5) - q5)) >
        Q5_LOCK_VERIFY_TOL)
      continue;
    save_J_sol(s2, s3, s4, s5, s6, k_E_O, r4, r6, r, Jsols, assembled_sols / 2,
               Jacobian_ee);
    if (joint_angles) {
      const array<double, 6> sol1 =
          q_from_axes_with_q5(s2, s3, s4, s5, s6, k_E_O, q5);
      const array<double, 3> sol2 = q_from_flipped_shoulder(sol1);
      const double q7 = PI / 4.0 - roots[i].delta;
      qsols[assembled_sols] = {sol1[0], sol1[1], sol1[2], sol1[3],
                               sol1[4], sol1[5], q7};
      check_limits(qsols[assembled_sols], 7, tuning);
      qsols[assembled_sols + 1] = {sol2[0],
                                   sol2[1],
                                   sol2[2],
                                   qsols[assembled_sols][3],
                                   qsols[assembled_sols][4],
                                   qsols[assembled_sols][5],
                                   qsols[assembled_sols][6]};
      check_limits(qsols[assembled_sols + 1], 3, tuning);
    }
    assembled_sols += 2;
  }
  return assembled_sols;
}

// FUNCTIONS FOR SWIVEL ANGLE (JACOBIAN)

static void franka_J_ik_q7_one_sol(
    const double q7, const array<double, 3>& i_E_O,
    const array<double, 3>& j_E_O, const array<double, 3>& k_E_O,
    array<double, 3>& i_6_O, const array<double, 3>& r_O7S_O,
    const array<double, 3>& r, array<array<array<double, 6>, 7>, 8>& Jsols,
    array<array<double, 7>, 8>& qsols, unsigned int ind,
    const bool joint_angles, const Frame Jacobian_ee, const unsigned int branch,
    const SolverTuning& tuning) {
  const double q1_sing = tuning.q1_sing;
  // returns the two solution related to one single branch of the IK with q7 as
  // free variable. The results are stored in Jsols[2*ind] and Jsols[2*ind+1]
  array<double, 3> s6;
  wrist_axes_from_q7(i_E_O, j_E_O, q7, i_6_O, s6);
  array<double, 3> r6 = {r_O7S_O[0] - a7 * i_6_O[0], r_O7S_O[1] - a7 * i_6_O[1],
                         r_O7S_O[2] - a7 * i_6_O[2]};
  double l = Norm(r6);
  double tmp = (b1 * b1 - l * l - b2 * b2) / (-2 * l * b2);
  if (!clamp_trig_roundoff(tmp)) {
    fill(qsols[2 * ind].begin(), qsols[2 * ind].end(), NAN);
    fill(qsols[2 * ind + 1].begin(), qsols[2 * ind + 1].end(), NAN);
    for (auto& row : Jsols[2 * ind]) fill(row.begin(), row.end(), NAN);
    for (auto& row : Jsols[2 * ind + 1]) fill(row.begin(), row.end(), NAN);
    return;
  }
  const double cos_actmp = tmp;
  const double sin_actmp = sqrt(std::max(0.0, 1.0 - cos_actmp * cos_actmp));
  array<double, 3> k_C_O = {-r6[0] / l, -r6[1] / l, -r6[2] / l};
  array<double, 3> i_C_O = Cross(k_C_O, s6);
  tmp = Norm(i_C_O);
  i_C_O = {i_C_O[0] / tmp, i_C_O[1] / tmp, i_C_O[2] / tmp};
  array<double, 3> j_C_O = Cross(k_C_O, i_C_O);
  double ry = s6[0] * j_C_O[0] + s6[1] * j_C_O[1] + s6[2] * j_C_O[2];
  double rz = s6[0] * k_C_O[0] + s6[1] * k_C_O[1] + s6[2] * k_C_O[2];
  const double sa2 = sin_beta2 * cos_actmp + cos_beta2 * sin_actmp;
  const double ca2 = cos_beta2 * cos_actmp - sin_beta2 * sin_actmp;
  tmp = -rz * ca2 / (ry * sa2);
  if (!clamp_trig_roundoff(tmp)) {
    fill(qsols[2 * ind].begin(), qsols[2 * ind].end(), NAN);
    fill(qsols[2 * ind + 1].begin(), qsols[2 * ind + 1].end(), NAN);
    for (auto& row : Jsols[2 * ind]) fill(row.begin(), row.end(), NAN);
    for (auto& row : Jsols[2 * ind + 1]) fill(row.begin(), row.end(), NAN);
    return;
  }
  const double sin_gamma = tmp;
  const double cos_gamma = sqrt(std::max(0.0, 1.0 - sin_gamma * sin_gamma));
  double v[3] = {-sa2 * cos_gamma, -sa2 * sin_gamma, -ca2};
  array<double, 3> s5;
  s5 = {i_C_O[0] * v[0] + j_C_O[0] * v[1] + k_C_O[0] * v[2],
        i_C_O[1] * v[0] + j_C_O[1] * v[1] + k_C_O[1] * v[2],
        i_C_O[2] * v[0] + j_C_O[2] * v[1] + k_C_O[2] * v[2]};
  if (branch == 1) {
    tmp = 2 * sa2 * cos_gamma;
    s5 = {s5[0] + tmp * i_C_O[0], s5[1] + tmp * i_C_O[1],
          s5[2] + tmp * i_C_O[2]};
  }
  array<double, 3> s4, r4, s3, s2;
  array<double, 6> sol1;
  array<double, 3> sol2;
  s4 = Cross(s5, r6);
  const double inverse_s4_norm = 1.0 / (l * fabs(sa2));
  s4 = {s4[0] * inverse_s4_norm, s4[1] * inverse_s4_norm,
        s4[2] * inverse_s4_norm};
  r4 = Cross(s5, s4);
  r4 = {r6[0] - d5 * s5[0] + a5 * r4[0], r6[1] - d5 * s5[1] + a5 * r4[1],
        r6[2] - d5 * s5[2] + a5 * r4[2]};
  rotate_by_beta1_scaled(s4, r4, s3);
  tmp = s3[1] * s3[1] + s3[0] * s3[0];
  if (tmp > SHOULDER_SING_TOL * SHOULDER_SING_TOL)
    s2 = shoulder_axis_from_s3(s3, tmp);
  else
    s2 = {sin(q1_sing), cos(q1_sing), 0};
  save_J_sol(s2, s3, s4, s5, s6, k_E_O, r4, r6, r, Jsols, ind, Jacobian_ee);
  if (joint_angles) {
    sol1 = q_from_axes(s2, s3, s4, s5, s6, k_E_O);
    sol2 = q_from_flipped_shoulder(sol1);
    qsols[2 * ind] = {sol1[0], sol1[1], sol1[2], sol1[3], sol1[4], sol1[5], q7};
    check_limits(qsols[2 * ind], 7, tuning);
    qsols[2 * ind + 1] = {sol2[0],           sol2[1],
                          sol2[2],           qsols[2 * ind][3],
                          qsols[2 * ind][4], qsols[2 * ind][5],
                          qsols[2 * ind][6]};
    check_limits(qsols[2 * ind + 1], 3, tuning);
  }
}

NANOGEOFIK_TARGET_CLONES
unsigned int franka_J_ik_swivel(const array<double, 3>& r,
                                const array<double, 9>& ROE, const double theta,
                                array<array<array<double, 6>, 7>, 8>& Jsols,
                                array<array<double, 7>, 8>& qsols,
                                const Output output, const Frame Jacobian_ee,
                                const SolverTuning& tuning) {
  const bool joint_angles = (output == Output::JointsAndJacobian);
  const double q1_sing = tuning.q1_sing;
  const unsigned int n_points = tuning.n_points;
  const unsigned int n_fine_search = tuning.n_fine_search;
  // IK to calculate Jacobian and joint angles with swivel angle as free
  // variable (numerical). INPUT: r = r_EO_O, position of frame E in frame O
  //        ROE, orientation of frame E in frame O (row-first format)
  //        theta, swivel angle (see paper for geometric definition)
  //        Jsols, array to store 8 Jacobian solutions
  //        qsols, array to store 8 joint-angle solutions
  //        joint_angles, if false only Jacobians are returned
  //        Jacobian_ee, end-effector frame of the Jacobian, not the IK. Only
  //        'E', 'F', '8' and '6' are supported. q1_sing, emergency value of q1
  //        in case of singularity at shoulder joints (type-1 singularity).
  //        n_points, number of points to discretize the range of q7
  // OUTPUT: number of solutions found.
  // NOTATION:
  // ri = r_iS_O,
  // si - s_i_O,
  if (n_points < 2 || n_points > MAX_N_POINTS) {
    for (unsigned int i = 0; i < qsols.size(); ++i) {
      fill(qsols[i].begin(), qsols[i].end(), NAN);
      for (auto& row : Jsols[i]) fill(row.begin(), row.end(), NAN);
    }
    return 0;
  }
  array<double, 3> k_E_O = {ROE[2], ROE[5], ROE[8]};
  // r_O7S_O = r_EO_O + r_OS_O + r_O7E_O = r_EO_O - (0,0,d1) - dE*k_E_O
  array<double, 3> r_O7S_O = {r[0] - dE * k_E_O[0], r[1] - dE * k_E_O[1],
                              r[2] - d1 - dE * k_E_O[2]};
  double tmp = Norm(r_O7S_O);
  if (tmp > b1 + b2 + a7) {
    for (int i = 0; i < 8; ++i) {
      fill(qsols[i].begin(), qsols[i].end(), NAN);
      for (auto& row : Jsols[i]) fill(row.begin(), row.end(), NAN);
    }
    return 0;
  }
  if (tmp < SING_TOL) {
    nanogeofik_log() << "ERROR: r_O7S_O is near zero";
    for (int i = 0; i < 8; ++i) {
      fill(qsols[i].begin(), qsols[i].end(), NAN);
      for (auto& row : Jsols[i]) fill(row.begin(), row.end(), NAN);
    }
    return 0;
  }
  array<double, 3> u_O7S_O = {r_O7S_O[0] / tmp, r_O7S_O[1] / tmp,
                              r_O7S_O[2] / tmp};
  array<double, 3> n1_O = stereographic_n1(u_O7S_O);
  const array<double, 3> sweep_i_E_O = {ROE[0], ROE[3], ROE[6]};
  const array<double, 3> sweep_j_E_O = {ROE[1], ROE[4], ROE[7]};
  array<double, 3> i_6_O;
  const SwivelGeometry geometry = build_swivel_geometry(
      sweep_i_E_O, sweep_j_E_O, n1_O, u_O7S_O, r_O7S_O, theta);
  SwivelSweep sweep;
  run_swivel_sweep(geometry, n_points, n_fine_search, tuning, sweep);

  const SwivelMinima& minima = sweep.minima;
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
    const double q7_opt = interpolate_swivel_q7(m[0], m[1], n_points,
                                                sweep.signed_errors, sweep.q7s);
    franka_J_ik_q7_one_sol(q7_opt, sweep_i_E_O, sweep_j_E_O, k_E_O, i_6_O,
                           r_O7S_O, r, Jsols, qsols, i, joint_angles,
                           Jacobian_ee, m[1], tuning);
  }
  for (int i = 2 * minima.count; i < 8; ++i) {
    for (auto& row : Jsols[i]) fill(row.begin(), row.end(), NAN);
  }
  for (int i = joint_angles ? 2 * minima.count : 0; i < 8; i++)
    fill(qsols[i].begin(), qsols[i].end(), NAN);
  return 2 * minima.count;
}
