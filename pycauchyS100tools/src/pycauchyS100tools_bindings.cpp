#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "pycauchyS100tools.h"

namespace py = pybind11;

PYBIND11_MODULE(libpycauchyS100tools, m)
{
    // 添加模块元信息
    m.attr("__version__") = "0.0.0";                                                  // 版本号
    m.attr("__author__") = "Cauchy - WuChao in D-Robotics";                           // 作者
    m.attr("__date__") = "2025-04-14";                                                // 日期
    m.attr("__doc__") = "pycauchyS100tools: This module provides an interface for BPU_ACTPolicy_VisionEncoder and BPU_ACTPolicy_TransformerLayers."; // 模块描述

    py::class_<BPU_ACTPolicy>(m, "BPU_ACTPolicy")
        .def(py::init<std::string &, std::string &>(), "init your BPU_ACTPolicy")
        .def("inference", &BPU_ACTPolicy::inference, "inference your input in np")
        .def("__call__", &BPU_ACTPolicy::inference, "__call__ is bind to inference");
}
