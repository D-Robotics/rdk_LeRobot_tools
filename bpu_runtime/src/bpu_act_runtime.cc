#include <pybind11/pybind11.h>
#include "bpu_pybind.hpp"

namespace py = pybind11;

PYBIND11_MODULE(bpu_act_runtime, m) {
    m.doc() = "C++ BPU ACT inference runtime for LeRobot (replaces hbm_runtime HB_HBMRuntime)";

    py::class_<BPUACTRuntime>(m, "BPUACTRuntime")
        .def(py::init<const std::vector<std::string>&>(),
             py::arg("model_paths"))
        .def("run", &BPUACTRuntime::run,
             py::arg("inputs"),
             py::arg("model_name"));
}
