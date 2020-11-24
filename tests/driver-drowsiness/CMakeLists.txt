set(SRCROOT ${PROJECT_SOURCE_DIR}/tests/driver-drowsiness)

# all source files
set(SRC
	${SRCROOT}/test_main.cpp
	${SRCROOT}/test_library.cpp
)

source_group("" FILES ${SRC})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

set(EXECUTABLE_NAME "${PROJECT_NAME}_test")
add_executable(${EXECUTABLE_NAME} ${SRC})
target_link_libraries(${EXECUTABLE_NAME} Library)