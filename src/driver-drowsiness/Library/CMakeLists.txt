set(INCROOT ${PROJECT_SOURCE_DIR}/include/driver-drowsiness/Library)
set(SRCROOT ${PROJECT_SOURCE_DIR}/src/driver-drowsiness/Library)

# all source files
file(GLOB INCS "${INCROOT}/*.hpp")
file(GLOB SRCS "${SRCROOT}/*.cpp")

set(CUSTOM_OUTPUT_DIR "${PROJECT_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CUSTOM_OUTPUT_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CUSTOM_OUTPUT_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CUSTOM_OUTPUT_DIR})

add_definitions(-DLibrary_EXPORTS)

add_library(Library SHARED ${SRCS} ${INCS})
