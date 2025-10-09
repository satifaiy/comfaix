cmake_minimum_required(VERSION 3.18)
project(ocr_test LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(TARGET ${PROJECT_NAME})

add_executable(${TARGET}
    paddle/paddle_recognize_test.cpp
)

# Link the test executable to the library and Google Test
target_link_libraries(${TARGET}
    PRIVATE
        pipeline
        segmentation
        ocr
        TBB::tbb
        openvino::runtime
        GTest::gtest_main
)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${TARGET} PRIVATE ${ASAN_FLAGS})
    target_link_options(${TARGET} PRIVATE ${ASAN_LINKER_FLAGS})
endif()

add_test(
  NAME ${TARGET}
  COMMAND $<TARGET_FILE:${TARGET}> WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
