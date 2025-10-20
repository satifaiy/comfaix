cmake_minimum_required(VERSION 3.18)
project(segmentation C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(TARGET ${PROJECT_NAME})

add_library(${TARGET} OBJECT
    src/cpp/text/paddle/config.cpp
    src/cpp/text/paddle/detect.cpp
    src/cpp/image/classic/detection.cpp
    src/cpp/image/classic/query.cpp
    src/cpp/image/classic/edge.cpp
    src/cpp/image/classic/edge_dexined.cpp
    src/cpp/image/classic/edge_teed.cpp
    src/cpp/image/classic/embed.cpp
    src/cpp/image/deeplearning/sam.cpp
    src/cpp/image/deeplearning/vit.cpp
    src/cpp/image/deeplearning/dino_v2.cpp
)

target_link_libraries(${TARGET} PUBLIC
    ${OpenCV_LIBRARIES}
    openvino::runtime
    pipeline
    simdjson
    Clipper2
)

add_library(${TARGET}_interface INTERFACE)

target_link_libraries(${TARGET}_interface INTERFACE
    ${OpenCV_LIBRARIES}
    openvino::runtime
    pipeline
    simdjson
    Clipper2
)

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cpp
)

if (COMFAIX_SELF_BUILD)
    add_subdirectory(src/cpp/test)
endif()
