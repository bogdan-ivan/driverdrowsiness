cmake_minimum_required(VERSION 2.8.12)
project(driver-drowsiness)

set(CMAKE_CXX_STANDARD 14)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/driver-drowsiness/common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/catch/include)

add_subdirectory(src/driver-drowsiness)
add_subdirectory(tests/driver-drowsiness)
