#include <opencv2/core/mat.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/dnn/utils/inference_engine.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "segmentation/image/classic/edge.hpp"

namespace segmentation::image::classic {

Dexined::Dexined(std::string onnx_path, std::string device) {
  net = cv::dnn::readNetFromONNX(onnx_path);
  net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

// @bried return detect opencv mat detected edge from the given src.
cv::Mat Dexined::detect(const cv::Mat &src) {
  // output blob is NCHW format
  cv::Mat blob = cv::dnn::blobFromImage(src, 1.0, cv::Size(512, 512),
                                        cv::Scalar(103.5, 116.2, 123.6), false,
                                        false, CV_32F);
  net.setInput(blob);

  // post processing
  int originalWidth = src.cols;
  int originalHeight = src.rows;

  std::vector<cv::Mat> outputs;
  net.forward(outputs);

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
    cv::resize(img, img, cv::Size(originalWidth, originalHeight));
    preds.push_back(img);
  }
  cv::Mat fuse = preds.back();
  cv::Mat ave = cv::Mat::zeros(originalHeight, originalWidth, CV_32F);
  for (cv::Mat &pred : preds) {
    cv::Mat temp;
    pred.convertTo(temp, CV_32F);
    ave += temp;
  }
  ave /= static_cast<float>(preds.size());
  ave.convertTo(ave, CV_8U);
  return fuse;
}

} // namespace segmentation::image::classic
