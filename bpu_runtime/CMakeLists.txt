cmake_minimum_required(VERSION 3.14)
project(bpu_act_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    /usr/hobot/include
    /usr/include/hobot/dnn
    /usr/include/hobot
)

link_directories(/usr/hobot/lib)

pybind11_add_module(bpu_act_runtime
    src/bpu_act_runtime.cc
)

target_link_libraries(bpu_act_runtime PRIVATE
    dnn
    hbucp
)
