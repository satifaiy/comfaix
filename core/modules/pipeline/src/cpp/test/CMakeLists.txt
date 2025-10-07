cmake_minimum_required(VERSION 3.18)
project(pipeline_test LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(TARGET ${PROJECT_NAME})

add_executable(${TARGET}
    image/batch_test.cpp
    image/chain_test.cpp
    image/permute_op_test.cpp
    image/resize_op_test.cpp
    image/segment_op_test.cpp
)

# Link the test executable to the library and Google Test
target_link_libraries(${TARGET}
    PRIVATE
        pipeline
        openvino::runtime
        TBB::tbb
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
