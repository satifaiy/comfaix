cmake_minimum_required(VERSION 3.24)

project(comfaix C CXX)

set(TARGET ${PROJECT_NAME})
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Setting root dir so sub directory can use it as relative access
# to file across project code. ${CMAKE_SOURCE_DIR} is not reliable
# as it's only valid when self build otherwise the directory is
# set to whosever include the project as library to their project.
set(ROOT_PROJECT_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(RELEASE_DIR       build/external/release)

# Setting variable self build to provide subdirectory
# a fine way to setting addition build based on whether
# it's self build or a part of other project's build.
set(COMFAIX_SELF_BUILD OFF)
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    set(COMFAIX_SELF_BUILD ON)
endif()

# defined build variable
set(BUILD_TEST COMFAIX_SELF_BUILD)

# Build dependencies only
# =======================================

if (BUILD_DEP_SHARE OR BUILD_DEP_STATIC)
    set(BUILD_TEST OFF)
    set(DEPENDENCIES_BUILD_SUFFIX "share")
    set(DEPENDENCIES_OUT_DIR ${ROOT_PROJECT_DIR}/${RELEASE_DIR}/dynamic)
    if (BUILD_DEP_STATIC)
        set(DEPENDENCIES_BUILD_SUFFIX "static")
        set(DEPENDENCIES_OUT_DIR ${ROOT_PROJECT_DIR}/${RELEASE_DIR}/static)
    endif()
    # Fetch all dependencies
    include(cmake/dependencies.cmake)
    return()
endif()

# Must perform BUILD dependencies first otherwise this include should failed
include(cmake/dependencies.prebuilt.cmake)

# Test Configuration
# ========================================

if (COMFAIX_SELF_BUILD)
    enable_testing()
    include(CTest)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug" AND COMFAIX_SELF_BUILD)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # Add ASan compiler and linker flags
        set(ASAN_FLAGS "-fsanitize=address")
        set(ASAN_LINKER_FLAGS "-fsanitize=address")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
        set(CTEST_MEMORYCHECK_TYPE "AddressSanitizer")
    endif()
endif()

# ========================================

# the modules will include all object and interface
set(COMFAIX_MODULES "")
add_subdirectory(modules)

set(OBJECTS "")
set(OBJECTS_DEPS "")
foreach(MODULE IN LISTS COMFAIX_MODULES)
    list(APPEND OBJECTS $<TARGET_OBJECTS:${MODULE}>)
    list(APPEND OBJECTS_DEPS "${MODULE}_interface")
endforeach()

add_library(${TARGET} SHARED ${OBJECTS})
target_link_libraries(${TARGET} PUBLIC
    ${OBJECTS_DEPS}
    TBB::tbb
)
