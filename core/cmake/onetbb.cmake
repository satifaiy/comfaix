# ******************* oneTBB ********************

set(ONETBB_DIR ${DEPENDENCIES_OUT_DIR}/oneTBB)
set(ONETBB_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/oneTBB-prefix/src/oneTBB-build-${DEPENDENCIES_BUILD_SUFFIX})

ExternalProject_Add(
  oneTBB_sanitize
  GIT_REPOSITORY    https://github.com/uxlfoundation/oneTBB.git
  GIT_TAG           v2022.2.0
  BUILD_IN_SOURCE   FALSE
  BINARY_DIR        ${ONETBB_BINARY_DIR}_sanitize
  INSTALL_DIR       ${ONETBB_DIR}_sanitize
  BUILD_COMMAND
    env TBB_ENABLE_SANITIZERS=1
    ${CMAKE_COMMAND} --build <BINARY_DIR> --config $<CONFIG> -- -j4
  CMAKE_ARGS
    --fresh
    -DTBB_TEST=OFF
    -DTBB_STRICT=ON
    -DCMAKE_INSTALL_PREFIX=${ONETBB_DIR}_sanitize
    -DBUILD_SHARED_LIBS=${BUILD_DEP_SHARE}
    -DCMAKE_BUILD_TYPE=Release
)

ExternalProject_Add(
  oneTBB
  GIT_REPOSITORY    https://github.com/uxlfoundation/oneTBB.git
  GIT_TAG           v2022.2.0
  BUILD_IN_SOURCE   FALSE
  BINARY_DIR        ${ONETBB_BINARY_DIR}
  INSTALL_DIR       ${ONETBB_DIR}
  CMAKE_ARGS
    --fresh
    -DTBB_TEST=OFF
    -DTBB_STRICT=ON
    -DCMAKE_INSTALL_PREFIX=${ONETBB_DIR}
    -DBUILD_SHARED_LIBS=${BUILD_DEP_SHARE}
    -DCMAKE_BUILD_TYPE=Release
)
