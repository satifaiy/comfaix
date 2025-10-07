# ******************* OpenVINO ********************

set(LIBVA_DIR ${DEPENDENCIES_OUT_DIR}/libva)
set(LIBVA_SHARE     enable-shared)
set(LIBVA_STATIC    disable-static)
# enable static
if (BUILD_DEP_STATIC OR NOT BUILD_DEP_SHARE)
    set(LIBVA_SHARE     disable-shared)
    set(LIBVA_STATIC    enable-static)
endif()

set(LIBVA_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/libva-prefix/src/libva-build-${DEPENDENCIES_BUILD_SUFFIX})

ExternalProject_Add(
  libva
  GIT_REPOSITORY      https://github.com/intel/libva.git
  GIT_TAG             2.22.0
  BUILD_ALWAYS        FALSE
  BUILD_IN_SOURCE     FALSE
  UPDATE_COMMAND      ""
  PREFIX              "${ROOT_PROJECT_DIR}/build/debug/libva-prefix"
  BINARY_DIR          ${LIBVA_BINARY_DIR}
  STAMP_DIR           "${ROOT_PROJECT_DIR}/build/debug/libva-prefix/src/libva-stamp"
  INSTALL_DIR         ${LIBVA_DIR}
  CONFIGURE_COMMAND   <SOURCE_DIR>/autogen.sh
    --${LIBVA_SHARE}
    --${LIBVA_STATIC}
    --prefix=${LIBVA_DIR}
    --libdir=${LIBVA_DIR}/lib
  BUILD_COMMAND $(MAKE)
  INSTALL_COMMAND $(MAKE) install
)

FetchContent_Declare(
    openvino_contrib
    GIT_REPOSITORY https://github.com/openvinotoolkit/openvino_contrib.git
    GIT_TAG        2022.3.0
)
FetchContent_MakeAvailable(openvino_contrib)

set(OPENVINO_DIR ${DEPENDENCIES_OUT_DIR}/openvino)
set(OPENVINO_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/openvino-prefix/src/openvino-build-${DEPENDENCIES_BUILD_SUFFIX})

set(OPENVINO_SRC_BIN ${CMAKE_CURRENT_BINARY_DIR}/openvino-prefix/src/openvino/bin)

# Delete openvino linking cache
add_custom_target(
    openvino_clean_bin
    COMMAND ${CMAKE_COMMAND} -E echo "Deleting stale staging directory: ${OPENVINO_SRC_BIN}"
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${OPENVINO_SRC_BIN}"
    COMMENT "Cleaning OpenVINO's staging 'bin' folder"
)

ExternalProject_Add(
  openvino
  GIT_REPOSITORY    https://github.com/openvinotoolkit/openvino.git
  GIT_TAG           2025.3.0
  BUILD_IN_SOURCE   FALSE
  BINARY_DIR        ${OPENVINO_BINARY_DIR}
  INSTALL_DIR       ${OPENVINO_DIR}
  DEPENDS           oneTBB
                    libva
                    openvino_clean_bin
  BUILD_COMMAND     ${CMAKE_COMMAND} --build <BINARY_DIR> --config $<CONFIG> -- -j4
  CMAKE_ARGS
    --fresh
    -DBUILD_SHARED_LIBS=${BUILD_DEP_SHARE}
    -DCMAKE_INSTALL_PREFIX=${OPENVINO_DIR}
    -DENABLE_TESTS=OFF
    -DENABLE_DOCS=OFF
    -DENABLE_WHEEL=OFF
    -DENABLE_PYTHON=OFF
    -DENABLE_SAMPLES=OFF
    -DENABLE_SYSTEM_PROTOBUF=OFF
    -DENABLE_SYSTEM_FLATBUFFERS=OFF
    -DENABLE_SYSTEM_TBB=ON
    -DENABLE_TBBBIND_2_5=OFF
    -DTBB_DIR=${ONETBB_DIR}/lib/cmake/TBB
    -DTBBROOT=${ONETBB_DIR}
    -DENABLE_SYSTEM_PUGIXML=OFF
    -DOPENVINO_EXTRA_MODULES=${openvino_contrib_SOURCE_DIR}
       -DBUILD_android_demos=OFF
       -DBUILD_java_api=OFF
       -DBUILD_nvidia_plugin=OFF
       -DBUILD_ollama_openvino=OFF
       -DBUILD_openvino-langchain=OFF
       -DBUILD_openvino_code=OFF
       -DBUILD_openvino_training_kit=OFF
    -DENABLE_OV_ONNX_FRONTEND=ON
    -DENABLE_OV_PADDLE_FRONTEND=ON
    -DENABLE_OV_TF_FRONTEND=ON
    -DENABLE_OV_TF_LITE_FRONTEND=ON
    -DENABLE_OV_PYTORCH_FRONTEND=ON
    -DENABLE_OV_JAX_FRONTEND=ON
    -DENABLE_OV_IR_FRONTEND=ON
    -DENABLE_PROXY=ON
    -DENABLE_AUTO_BATCH=ON
    -DENABLE_TEMPLATE=ON
    -DENABLE_AUTO=ON
    -DENABLE_MULTI=ON
    -DENABLE_HETERO=ON
    -DENABLE_INTEL_GPU=ON
    -DENABLE_INTEL_CPU=ON
    -DENABLE_INTEL_NPU=OFF
    -DENABLE_PROFILING_ITT=OFF
    -DENABLE_PROFILING_FIRST_INFERENCE=OFF
    -DENABLE_JS=OFF
    -DENABLE_SSE42=ON
    -DENABLE_AVX2=ON
    -DENABLE_AVX512F=ON
    -DSELECTIVE_BUILD=OFF
    -DENABLE_MLAS_FOR_CPU=ON
    -DCMAKE_PREFIX_PATH=${LIBVA_DIR}/pkgconfig
    -DCMAKE_BUILD_TYPE=Release
)
