cmake_minimum_required(VERSION 3.0)

include_directories(
    include
)

find_package(pybind11 REQUIRED)

SET(LINK_libs dnn ${os_base_lib})

add_library(pycauchyX5tools MODULE 
            src/pycauchyX5tools_bindings.cpp 
            src/pycauchyX5tools.cpp)
target_link_libraries(pycauchyX5tools PRIVATE pybind11::module ${LINK_libs})