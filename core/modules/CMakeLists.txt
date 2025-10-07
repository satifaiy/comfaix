cmake_minimum_required(VERSION 3.18)
project(modules C CXX)

set(TARGET ${PROJECT_NAME})

add_library(${TARGET} INTERFACE)

set(COMFAIX_MODULES
    pipeline
    segmentation
    # ocr
)

# update core variable
set(COMFAIX_MODULES
    ${COMFAIX_MODULES}
    PARENT_SCOPE
)

add_subdirectory(common)
# loop add all modules
message(STATUS "Add Modules")
foreach(MODULE IN LISTS COMFAIX_MODULES)
    add_subdirectory(${MODULE})
    message(STATUS "  ${MODULE}")
endforeach()
