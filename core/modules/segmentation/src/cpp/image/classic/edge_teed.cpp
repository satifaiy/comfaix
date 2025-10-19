#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/dnn/utils/inference_engine.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/runtime/core.hpp>

#include "segmentation/image/classic/edge.hpp"

namespace segmentation::image::classic {

Teed::Teed(std::string onnx_path, std::string device) {
  net = cv::dnn::readNetFromONNX(onnx_path);
  net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

// @bried return detect opencv mat detected edge from the given src.
cv::Mat Teed::detect(const cv::Mat &src) {
  cv::Mat input = src;
  cv::Mat target = input;
  if (input.rows < 512 || input.cols < 512)
    cv::resize(target, input, cv::Size(0, 0), 1.5, 1.5);

  if (input.rows % 8 != 0 || input.cols % 8 != 0) {
    target = input;
    auto img_width = ((src.rows / 8) + 1) * 8;
    auto img_height = ((src.cols / 8) + 1) * 8;
    cv::resize(target, input, cv::Size(img_width, img_height));
  }

  // output blob is NCHW format
  cv::Mat blob =
      cv::dnn::blobFromImage(input, 1.0, cv::Size(512, 512),
                             {104.007, 116.669, 122.679}, false, false, CV_32F);
  net.setInput(blob);

  std::vector<cv::Mat> outputs;
  net.forward(outputs);

  // post processing
  int originalWidth = src.cols;
  int originalHeight = src.rows;

  std::vector<cv::Mat> preds;
  preds.reserve(outputs.size());
  for (const cv::Mat &p : outputs) {
    cv::Mat img;
    cv::Mat processed;
    if (p.dims == 4 && p.size[0] == 1 && p.size[1] == 1) {
      processed = p.reshape(0, {p.size[2], p.size[3]});
    } else {
      processed = p.clone();
    }
    cv::exp(-processed, processed); // e^-input
    processed = 1.0 / (1.0 + processed);
    cv::normalize(processed, img, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::bitwise_not(processed, processed);
    cv::resize(img, img, cv::Size(originalWidth, originalHeight));
    preds.push_back(img);
  }
  return preds.back();
}

} // namespace segmentation::image::classic
