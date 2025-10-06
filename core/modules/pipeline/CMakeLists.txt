cmake_minimum_required(VERSION 3.18)
project(pipeline LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(TARGET ${PROJECT_NAME})

add_library(${TARGET} OBJECT
    src/cpp/image/score.cpp
    src/cpp/image/segment_op.cpp
)

target_link_libraries(${TARGET} PUBLIC
    ${OpenCV_LIBS}
    common
)

# propagate
add_library(${TARGET}_interface INTERFACE)

target_link_libraries(${TARGET}_interface INTERFACE
    ${OpenCV_LIBS}
    common
)

target_include_directories(${TARGET} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/cpp>
)

if (COMFAIX_SELF_BUILD)
    add_subdirectory(src/cpp/test)
endif()
