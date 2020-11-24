#set(INCROOT ${PROJECT_SOURCE_DIR}/include/Spring/Loader)
set(SRCROOT ${PROJECT_SOURCE_DIR}/src/driver-drowsiness/Loader)

# all source files
set(SRC
    ${SRCROOT}/main.cpp
)

source_group("" FILES ${SRC})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

include_directories("${PROJECT_SOURCE_DIR}/bin")

add_executable(Loader ${SRC})

target_link_libraries(Loader Library)