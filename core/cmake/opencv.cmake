# ******************* OpenCV ********************

FetchContent_Declare(
    opencv_contrib
    GIT_REPOSITORY https://github.com/opencv/opencv_contrib.git
    GIT_TAG        4.12.0
)
FetchContent_MakeAvailable(opencv_contrib)

set(OPENCV_DIR ${DEPENDENCIES_OUT_DIR}/opencv)
set(OPENCV_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/opencv-prefix/src/opencv-build-${DEPENDENCIES_BUILD_SUFFIX})

ExternalProject_Add(
  opencv
  GIT_REPOSITORY    https://github.com/opencv/opencv.git
  GIT_TAG           4.12.0
  BUILD_IN_SOURCE   FALSE
  INSTALL_DIR       ${OPENCV_DIR}
  BINARY_DIR        ${OPENCV_BINARY_DIR}
  DEPENDS           oneTBB
                    openvino
  BUILD_COMMAND     ${CMAKE_COMMAND} --build <BINARY_DIR> --config $<CONFIG> -- -j4
  CMAKE_ARGS
    --fresh
    -DOPENCV_EXTRA_MODULES_PATH=${opencv_contrib_SOURCE_DIR}/modules
        -DBUILD_EXAMPLES=OFF
        -DBUILD_opencv_julia=OFF
        -DBUILD_opencv_matlab=OFF
        -DBUILD_opencv_mcc=OFF
        -DBUILD_opencv_optflow=OFF
        -DBUILD_opencv_ovis=OFF
        -DBUILD_opencv_phase_unwrapping=OFF
        -DBUILD_opencv_plot=OFF
        -DBUILD_opencv_rapid=OFF
        -DBUILD_opencv_reg=OFF
        -DBUILD_opencv_rgbd=OFF
        -DBUILD_opencv_saliency=OFF
        -DBUILD_opencv_signal=OFF
        -DBUILD_opencv_structured_light=OFF
        -DBUILD_opencv_viz=OFF
        -DBUILD_opencv_wechat_qrcode=OFF
        -DBUILD_opencv_xphoto=OFF
        -DBUILD_opencv_fuzzy=OFF
        -DBUILD_opencv_freetype=OFF
        -DBUILD_opencv_cvv=OFF
    -DBUILD_SHARED_LIBS=${BUILD_DEP_SHARE}
    -DBUILD_EXAMPLES=OFF
    -DBUILD_TESTS=OFF
    -DBUILD_PERF_TESTS=OFF
    -DBUILD_JASPER=OFF
    -DWITH_JASPER=OFF
    -DBUILD_JAVA=OFF
    -DBUILD_opencv_java=OFF
    -DWITH_1394=OFF
    -DWITH_CUDA=OFF
    -DWITH_EIGEN=OFF
    -DWITH_GPHOTO2=OFF
    -DWITH_GTK_2_X=OFF
    -DWITH_LAPACK=OFF
    -DWITH_MATLAB=OFF
    -DENABLE_PRECOMPILED_HEADERS=OFF
    -DINSTALL_TESTS=OFF
    -DINSTALL_C_EXAMPLES=OFF
    -DINSTALL_PYTHON_EXAMPLES=OFF
    -DINSTALL_PDB=OFF
    -DBUILD_opencv_python2=OFF
    -DBUILD_opencv_python3=OFF
    -DPYTHON3_PACKAGES_PATH=install/python/python3
    -DPYTHON3_LIMITED_API=ON
    -DOPENCV_PYTHON_INSTALL_PATH=python
    -DBUILD_OPENEXR=ON
    -DWITH_OPENEXR=ON
    -DBUILD_APPS_LIST=version
    -DBUILD_INFO_SKIP_EXTRA_MODULES=ON
    -DBUILD_JPEG=ON
    -DBUILD_PNG=ON
    -DBUILD_WEBP=ON
    -DWITH_WEBP=ON
    -DWITH_OPENJPEG=ON
    -DBUILD_ZLIB=ON
    -DWITH_GSTREAMER=OFF
    -DOPENCV_GAPI_GSTREAMER=OFF
    -DBUILD_opencv_apps=ON
    -DWITH_IPP=ON
    -DWITH_MFX=ON
    -DWITH_OPENCL=ON
    -DWITH_OPENCLAMDFFT=OFF
    -DWITH_OPENCLAMDBLAS=OFF
    -DWITH_QUIRC=ON
    -DBUILD_TBB=OFF
    -DWITH_TBB=ON
    -DMKL_WITH_TBB=ON
    -DTBBROOT=${DEPENDENCIES_OUT_DIR}/oneTBB
    -DTBB_DIR=${DEPENDENCIES_OUT_DIR}/oneTBB/lib/cmake/TBB
    -DWITH_TIFF=OFF
    -DWITH_VTK=OFF
    -DCMAKE_USE_RELATIVE_PATHS=ON
    -DCMAKE_SKIP_INSTALL_RPATH=ON
    -DENABLE_BUILD_HARDENING=ON
    -DENABLE_CONFIG_VERIFICATION=ON
    -DENABLE_CXX11=ON
    -DHIGHGUI_ENABLE_PLUGINS=OFF
    -DWITH_GTK=OFF
    -DWITH_WIN32UI=OFF
    -DWITH_QT=OFF
    -DWITH_FRAMEBUFFER=OFF
    -DWITH_FRAMEBUFFER_XVFB=OFF
    -DCMAKE_INSTALL_PREFIX=${OPENCV_DIR}
    -DOPENCV_SKIP_PKGCONFIG_GENERATION=OFF
    -DOPENCV_SKIP_PYTHON_LOADER=ON
    -DOPENCV_SKIP_CMAKE_ROOT_CONFIG=ON
    -DOPENCV_GENERATE_SETUPVARS=OFF
    -DOPENCV_BIN_INSTALL_PATH=bin
    -DOPENCV_INCLUDE_INSTALL_PATH=include
    -DOPENCV_LIB_INSTALL_PATH=lib
    -DOPENCV_CONFIG_INSTALL_PATH=cmake
    -DOPENCV_3P_LIB_INSTALL_PATH=3rdparty
    -DOPENCV_SAMPLES_SRC_INSTALL_PATH=samples
    -DOPENCV_DOC_INSTALL_PATH=doc
    -DOPENCV_OTHER_INSTALL_PATH=etc
    -DOPENCV_LICENSES_INSTALL_PATH=etc/licenses
    -DOPENCV_INSTALL_FFMPEG_DOWNLOAD_SCRIPT=ON
    -DBUILD_opencv_world=ON
    -DCPU_BASELINE=SSE4_2
    -DOPENCV_IPP_GAUSSIAN_BLUR=ON
    -DWITH_OPENVINO=ON
    -DOpenVINO_DIR=${DEPENDENCIES_OUT_DIR}/openvino/runtime/cmake
    -DVIDEOIO_PLUGIN_LIST=ffmpeg,gstreamer,mfx
    -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined
    -DCMAKE_BUILD_TYPE=Release
)

message(STATUS ${DEPENDENCIES_OUT_DIR}/openvino/runtime/cmake)
