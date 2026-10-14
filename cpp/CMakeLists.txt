cmake_minimum_required(VERSION 3.20)
project(thermal_production LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(thermal_device src/device.cpp)
target_include_directories(thermal_device PUBLIC include)

add_executable(thermal_production_test src/production_test.cpp)
target_link_libraries(thermal_production_test PRIVATE thermal_device)
target_compile_options(thermal_production_test PRIVATE -Wall -Wextra -O2)

enable_testing()
add_test(NAME production COMMAND thermal_production_test ${CMAKE_SOURCE_DIR}/..)
