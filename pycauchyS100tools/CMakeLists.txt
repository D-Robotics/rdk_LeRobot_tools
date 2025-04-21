cmake_minimum_required(VERSION 3.0)

include_directories(
    include
    /usr/include/hobot
)

link_directories(
    /usr/local/lib
    /usr/lib/aarch64-linux-gnu
    /usr/hobot/lib
)

find_package(pybind11 REQUIRED)

SET(LINK_libs hbucp dnn ${os_base_lib})

add_library(pycauchyS100tools MODULE 
            src/pycauchyS100tools_bindings.cpp 
            src/pycauchyS100tools.cpp)
target_link_libraries(pycauchyS100tools PRIVATE pybind11::module ${LINK_libs})
