#pragma once

#include <Eigen/Geometry>
#include <array>
#include <unsupported/Eigen/EulerAngles>
#include <variant>

namespace franky {

using Vector6d = Eigen::Vector<double, 6>;
using Vector7d = Eigen::Vector<double, 7>;
using Jacobian = Eigen::Matrix<double, 6, 7>;
using IntertiaMatrix = Eigen::Matrix<double, 3, 3>;

using Affine = Eigen::Affine3d;

template <size_t dims>
using Array = std::variant<std::array<double, dims>, Eigen::Vector<double, dims>>;

template <size_t dims>
using ScalarOrArray = std::variant<double, Array<dims>>;

}  // namespace franky
