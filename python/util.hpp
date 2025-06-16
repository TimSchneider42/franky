#pragma once

#include <iostream>

using namespace franky;

template <typename T>
std::string strFromStream(const T &obj) {
  std::stringstream ss;
  ss << obj;
  return ss.str();
}
