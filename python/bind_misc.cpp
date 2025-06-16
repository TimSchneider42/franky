#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "franky.hpp"
#include "util.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the '_a' literal
using namespace franky;

void bind_misc(py::module &m) {
  py::class_<franka::Duration>(m, "Duration")
      .def(py::init<>())
      .def(py::init<uint64_t>())
      .def("to_sec", &franka::Duration::toSec)
      .def("to_msec", &franka::Duration::toMSec)
      .def(py::self + py::self)
      .def(py::self += py::self)
      .def(py::self - py::self)
      .def(py::self -= py::self)
      .def(py::self * uint64_t())
      .def(py::self *= uint64_t())
      .def(py::self / uint64_t())
      .def(py::self /= uint64_t())
      .def("__repr__", strFromStream<franka::Duration>)
      .def(
          py::pickle(
              [](const franka::Duration &duration) {  // __getstate__
                return py::make_tuple(duration.toMSec());
              },
              [](const py::tuple &t) {  // __setstate__
                if (t.size() != 1) throw std::runtime_error("Invalid state!");
                return franka::Duration(t[0].cast<uint64_t>());
              }));

  py::class_<RelativeDynamicsFactor>(m, "RelativeDynamicsFactor")
      .def(py::init<>())
      .def(py::init<double>(), "value"_a)
      .def(py::init<double, double, double>(), "velocity"_a, "acceleration"_a, "jerk"_a)
      .def_property_readonly("velocity", &RelativeDynamicsFactor::velocity)
      .def_property_readonly("acceleration", &RelativeDynamicsFactor::acceleration)
      .def_property_readonly("jerk", &RelativeDynamicsFactor::jerk)
      .def_property_readonly("max_dynamics", &RelativeDynamicsFactor::max_dynamics)
      .def_property_readonly_static("MAX_DYNAMICS", [](py::object) { return RelativeDynamicsFactor::MAX_DYNAMICS(); })
      .def(py::self * py::self);
  py::implicitly_convertible<double, RelativeDynamicsFactor>();

  py::class_<std::shared_future<bool>>(m, "BoolFuture")
      .def(
          "wait",
          [](const std::shared_future<bool> &future, std::optional<double> timeout) {
            if (timeout.has_value())
              return future.wait_for(std::chrono::duration<double>(timeout.value())) == std::future_status::ready;
            future.wait();
            return true;
          },
          "timeout"_a = std::nullopt,
          py::call_guard<py::gil_scoped_release>())
      .def("get", &std::shared_future<bool>::get, py::call_guard<py::gil_scoped_release>());
}
