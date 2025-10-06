cmake_minimum_required(VERSION 3.18)
project(segmentation C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(TARGET ${PROJECT_NAME})

add_library(${TARGET} OBJECT
    src/text/paddle/detect.cpp
)

# target_link_libraries(${TARGET} PUBLIC
#     ${OpenCV_LIBRARIES}
#     openvino::runtime
#     pipeline
# )

add_library(${TARGET}_interface INTERFACE)

# target_link_libraries(${TARGET}_interface INTERFACE
#     ${OpenCV_LIBRARIES}
#     openvino::runtime
#     pipeline
# )

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/c
)
