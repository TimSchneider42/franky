#include <franky/types.hpp>

franky::Affine mk_affine(double x, double y, double z, double a = 0, double b = 0, double c = 0) {
  return franky::Affine().fromPositionOrientationScale(
      Eigen::Vector3d({x, y, z}),
      Eigen::EulerAngles<double, Eigen::EulerSystemXYZ>({a, b, c}),
      Eigen::Vector3d::Ones());
}
