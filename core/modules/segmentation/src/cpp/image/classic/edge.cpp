#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include "segmentation/image/classic/edge.hpp"

namespace segmentation::image::classic {

// @brief create sober edge detection
EdgeDetection sobel(int kernel_width, int kernel_height) {
  return [kernel_width, kernel_height](const cv::Mat &src) -> cv::Mat {
    cv::Mat matx, maty;
    cv::Sobel(src, matx, CV_32F, 1, 0, kernel_width);
    cv::Sobel(src, maty, CV_32F, 0, 1, kernel_height);
    cv::Mat magnitude;
    cv::magnitude(matx, maty, magnitude);
    cv::convertScaleAbs(magnitude, magnitude);
    return magnitude;
  };
}

// @brief create canny edge detection
EdgeDetection canny(double threshold1, double threshold2, int apertureSize,
                    bool L2gradient) {
  return [threshold1, threshold2, apertureSize,
          L2gradient](const cv::Mat &src) -> cv::Mat {
    cv::Mat out;
    cv::Mat gray = src;
    if (src.type() != CV_8UC1)
      cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, out, threshold1, threshold2, apertureSize, L2gradient);
    return out;
  };
}

// @brief create dexined edge detection
EdgeDetection dexined(std::string onnx_path, std::string device) {
  std::unique_ptr<Dexined> dxn = std::make_unique<Dexined>(onnx_path, device);
  auto state = std::make_shared<std::unique_ptr<Dexined>>(std::move(dxn));
  return
      [state](const cv::Mat &src) -> cv::Mat { return (**state).detect(src); };
}

// @brief create teed edge detection
EdgeDetection teed(std::string onnx_path, std::string device) {
  std::unique_ptr<Teed> teed = std::make_unique<Teed>(onnx_path, device);
  auto state = std::make_shared<std::unique_ptr<Teed>>(std::move(teed));
  return
      [state](const cv::Mat &src) -> cv::Mat { return (**state).detect(src); };
}

} // namespace segmentation::image::classic
