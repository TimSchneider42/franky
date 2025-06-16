#pragma once

#include <franka/duration.h>

#include <Eigen/Core>
#include <array>

namespace franky {

template <typename T, size_t dims>
std::array<T, dims> toStd(const Eigen::Matrix<T, dims, 1> &vector) {
  std::array<T, dims> result;
  Eigen::Matrix<T, dims, 1>::Map(result.data()) = vector;
  return result;
}

template <size_t dims>
std::array<double, dims> toStdD(const Eigen::Matrix<double, dims, 1> &vector) {
  return toStd<double, dims>(vector);
}

template <typename T, size_t dims>
Eigen::Matrix<T, dims, 1> toEigen(const std::array<T, dims> &vector) {
  return Eigen::Matrix<T, dims, 1>::Map(vector.data());
}

template <size_t dims>
Eigen::Matrix<double, dims, 1> toEigenD(const std::array<double, dims> &vector) {
  return toEigen<double, dims>(vector);
}

template <size_t rows, size_t cols>
std::array<double, rows * cols> toStdDMatD(const Eigen::Matrix<double, rows, cols, Eigen::ColMajor> &matrix) {
  std::array<double, rows * cols> result;
  Eigen::Map<Eigen::Matrix<double, rows, cols, Eigen::ColMajor>>(result.data()) = matrix;
  return result;
}

template <size_t rows, size_t cols>
Eigen::Matrix<double, rows, cols, Eigen::ColMajor> toEigenMatD(const std::array<double, rows * cols> &array) {
  return Eigen::Map<const Eigen::Matrix<double, rows, cols, Eigen::ColMajor>>(array.data());
}

inline Affine stdToAffine(const std::array<double, 16> &array) {
  Affine result;
  result.matrix() = toEigenMatD<4, 4>(array);
  return result;
}

template <size_t dims>
Eigen::Vector<double, dims> ensureEigen(const Array<dims> &input) {
  if (std::holds_alternative<Eigen::Vector<double, dims>>(input)) return std::get<Eigen::Vector<double, dims>>(input);
  return toEigenD<dims>(std::get<std::array<double, dims>>(input));
}

template <size_t dims>
std::array<double, dims> ensureStd(const Array<dims> &input) {
  if (std::holds_alternative<std::array<double, dims>>(input)) return std::get<std::array<double, dims>>(input);
  return toStdD<dims>(std::get<Eigen::Vector<double, dims>>(input));
}

template <size_t dims>
std::array<double, dims> expand(const ScalarOrArray<dims> &input) {
  if (std::holds_alternative<Array<dims>>(input)) {
    return ensureStd<dims>(std::get<Array<dims>>(input));
  }
  std::array<double, dims> output;
  std::fill(output.begin(), output.end(), std::get<double>(input));
  return output;
}

template <size_t dims>
Eigen::Vector<double, dims> expandEigen(const ScalarOrArray<dims> &input) {
  if (std::holds_alternative<Array<dims>>(input)) {
    return ensureEigen<dims>(std::get<Array<dims>>(input));
  }
  Eigen::Vector<double, dims> output;
  output.fill(std::get<double>(input));
  return output;
}

template <int dims>
std::ostream &operator<<(std::ostream &os, const Eigen::Vector<double, dims> &vec) {
  os << "[";
  for (size_t i = 0; i < dims; i++) {
    os << vec[i];
    if (i != dims - 1) os << " ";
  }
  os << "]";
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const Affine &affine) {
  os << "Affine(t=" << affine.translation().eval() << ", q=" << Eigen::Quaterniond(affine.rotation()).coeffs() << ")";
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const franka::Duration &duration) {
  os << "Duration(" << duration.toMSec() << ")";
  return os;
}

}  // namespace franky
