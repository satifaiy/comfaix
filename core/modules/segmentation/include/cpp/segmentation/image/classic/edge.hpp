#pragma once

#include <functional>
#include <opencv2/core/mat.hpp>
#include <opencv2/dnn.hpp>
#include <openvino/core/model.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/infer_request.hpp>

namespace segmentation::image::classic {

// @brief edge detection provide method to detect the edge.
using EdgeDetection = std::function<cv::Mat(const cv::Mat &src)>;

// @brief create sober edge detection
EdgeDetection sobel(int kernel_width, int kernel_height);

// @brief create canny edge detection
EdgeDetection canny(double threshold1, double threshold2, int apertureSize = 3,
                    bool L2gradient = false);

// @brief create dexined edge detection
EdgeDetection dexined(std::string onnx_path, std::string device);

// @brief dexined class providing loading model using openvino
class Dexined {
public:
  Dexined(std::string onnx_path, std::string device);

  // @bried return detect opencv mat detected edge from the given src.
  cv::Mat detect(const cv::Mat &src);

private:
  cv::dnn::Net net;
};

// @brief create teed edge detection
EdgeDetection teed(std::string onnx_path, std::string device);

// @brief create teed edge detection
class Teed {
public:
  Teed(std::string onnx_path, std::string device);

  // @bried return detect opencv mat detected edge from the given src.
  cv::Mat detect(const cv::Mat &src);

private:
  cv::dnn::Net net;
};

} // namespace segmentation::image::classic
