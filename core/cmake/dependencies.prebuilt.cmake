# It's expected that the dependencies is pre-built
# ================================================

set(BUILD_TYPE "dynamic")
if (BUILD_SINGLE_PKG)
    # static linking into a single .so file
    # or if source code is built as static
    set(BUILD_TYPE "static")
endif()

set(LIB_DIR ${ROOT_PROJECT_DIR}/${RELEASE_DIR})
set(CMAKE_PREFIX_PATH
        "${LIB_DIR}/${BUILD_TYPE}/libva/lib/pkgconfig"
        "${LIB_DIR}/${BUILD_TYPE}/oneTBB/lib/cmake/TBB"
        "${LIB_DIR}/${BUILD_TYPE}/opencv/cmake"
        "${LIB_DIR}/${BUILD_TYPE}/openvino/runtime/cmake"
)

message(STATUS "--- DEPENDENCIES directory ---")
foreach(DEP_PATH IN LISTS CMAKE_PREFIX_PATH)
    # The message() command is called once for each item
    message(STATUS "  - ${DEP_PATH}")
endforeach()
message(STATUS "------------------------------")

find_package(OpenCV CONFIG REQUIRED)
find_package(OpenVINO CONFIG REQUIRED)
find_package(TBB REQUIRED)

include(FetchContent)
if (BUILD_TEST)
    include(${CMAKE_CURRENT_LIST_DIR}/googletest.cmake)
endif()
