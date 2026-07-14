#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "docstrings.hpp"
#include "franky.hpp"
#include "util.hpp"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the '_a' literal
using namespace franky;

void bind_misc(py::module &m) {
  py::class_<franka::Duration>(m, "Duration", DOC(franka, Duration))
      .def(py::init<>(), DOC(franka, Duration, Duration))
      .def(py::init<uint64_t>(), DOC(franka, Duration, Duration_2))
      .def("to_sec", &franka::Duration::toSec, DOC(franka, Duration, toSec))
      .def("to_msec", &franka::Duration::toMSec, DOC(franka, Duration, toMSec))
      .def(py::self + py::self)
      .def(py::self += py::self)
      .def(py::self - py::self)
      .def(py::self -= py::self)
      .def(py::self * uint64_t())
      .def(py::self *= uint64_t())
      .def(
          "__truediv__",
          [](const franka::Duration &duration, uint64_t divisor) {
            if (divisor == 0) {
              PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
              throw py::error_already_set();
            }
            return duration / divisor;
          },
          py::is_operator())
      .def(
          "__itruediv__",
          [](franka::Duration &duration, uint64_t divisor) -> franka::Duration & {
            if (divisor == 0) {
              PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
              throw py::error_already_set();
            }
            duration /= divisor;
            return duration;
          },
          py::return_value_policy::reference_internal,
          py::is_operator())
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

  py::class_<RelativeDynamicsFactor>(m, "RelativeDynamicsFactor", DOC(franky, RelativeDynamicsFactor))
      .def(py::init<>(), DOC(franky, RelativeDynamicsFactor, RelativeDynamicsFactor))
      .def(py::init<double>(), "value"_a, DOC(franky, RelativeDynamicsFactor, RelativeDynamicsFactor_2))
      .def(
          py::init<double, double, double>(),
          "velocity"_a,
          "acceleration"_a,
          "jerk"_a,
          DOC(franky, RelativeDynamicsFactor, RelativeDynamicsFactor_3))
      .def_property_readonly(
          "velocity", &RelativeDynamicsFactor::velocity, DOC(franky, RelativeDynamicsFactor, velocity))
      .def_property_readonly(
          "acceleration", &RelativeDynamicsFactor::acceleration, DOC(franky, RelativeDynamicsFactor, acceleration))
      .def_property_readonly("jerk", &RelativeDynamicsFactor::jerk, DOC(franky, RelativeDynamicsFactor, jerk))
      .def_property_readonly(
          "max_dynamics", &RelativeDynamicsFactor::max_dynamics, DOC(franky, RelativeDynamicsFactor, max_dynamics))
      .def_property_readonly_static(
          "MAX_DYNAMICS",
          [](py::object) { return RelativeDynamicsFactor::MAX_DYNAMICS(); },
          DOC(franky, RelativeDynamicsFactor, MAX_DYNAMICS))
      .def(py::self * py::self);
  py::implicitly_convertible<double, RelativeDynamicsFactor>();

  py::class_<std::shared_future<bool>>(
      m,
      "BoolFuture",
      "A future that will eventually contain a boolean result, e.g. of an asynchronous gripper command.")
      .def(
          "wait",
          [](const std::shared_future<bool> &future, std::optional<double> timeout) {
            return waitForFutureInterruptibly(future, timeout);
          },
          "timeout"_a = std::nullopt,
          py::call_guard<py::gil_scoped_release>(),
          "Wait for the result to become available.\n\n"
          "Args:\n"
          "    timeout: Maximum time to wait [s]. If None, waits indefinitely.\n\n"
          "Returns:\n"
          "    True if the result became available before the timeout expired, False otherwise.")
      .def(
          "get",
          [](const std::shared_future<bool> &future) {
            waitForFutureInterruptibly(future);
            return future.get();
          },
          py::call_guard<py::gil_scoped_release>(),
          "Wait for the result and return it. Blocks until the result is available.");
}
