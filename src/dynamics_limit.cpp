#include "franky/dynamics_limit.hpp"

#include <array>

#include "franky/types.hpp"
#include "franky/util.hpp"

namespace franky {

template <>
void DynamicsLimit<double>::check(const double &value) const {
  if (value < 0) {
    std::stringstream ss;
    ss << desc << " limit cannot be negative.";
    throw std::runtime_error(ss.str());
  }
  if (value > max) {
    std::stringstream ss;
    ss << desc << " limit (" << value << ") cannot be greater than the maximum value (" << max << ").";
    throw std::runtime_error(ss.str());
  }
}

template <>
template <>
void DynamicsLimit<Vector7d>::setFrom<Array<7>>(const Array<7> &value) {
  set(ensureEigen<7>(value));
}

template <>
void DynamicsLimit<Vector7d>::check(const Vector7d &value) const {
  for (int i = 0; i < value.size(); i++) {
    if (value[i] < 0) {
      std::stringstream ss;
      ss << desc << " limit cannot be negative.";
      throw std::runtime_error(ss.str());
    }
    if (value[i] > max[i]) {
      std::stringstream ss;
      ss << desc << " limit at index " << i << "(" << value[i] << ") cannot be greater than the maximum value ("
         << max[i] << ").";
      throw std::runtime_error(ss.str());
    }
  }
}

std::ostream &operator<<(std::ostream &os, const DynamicsLimit<Vector7d> &limit) {
  os << limit.get() << " (max: " << Vector7d::Map(limit.max.data()).eval() << ")";
  return os;
}

}  // namespace franky
