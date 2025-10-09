#pragma once

#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include "segmentation/segment.hpp"
#include "segmentation/text/paddle/config.hpp"

namespace segmentation::text::paddle {

// @brief a detection class use to detect text from an image.
class Detection
    : public segmentation::ISegmentation<cv::Mat, std::vector<cv::Point2f>> {
public:
  // @brief create Detect with the given model path and device.
  // @param path is the path to the text detection model.
  // @param device is the openvino backend
  Detection(std::string path, std::string device);

  // @brief create Detect with the given model config and device.
  // @param detect model configuration
  // @param device is the openvino backend
  Detection(const DetectConfig &config);

  // @brief implement ISegementation.
  std::vector<common::ImageSegmentionResult> segment(const cv::Mat &m) override;

  // @brief provide simple consistent sort order.
  // @detail the sort is group by rows and then by columns. If the segment is a
  // row span or column span the segment will add to earliest row or columns.
  // To have procise location, use segmentation::GridVisualization which layout
  // the exact location of the detected text on the image. It's also providing
  // support for both column and row span.
  static void sort_results(std::vector<common::ImageSegmentionResult> &results,
                           float tolerance = 0);

private:
  const DetectConfig config;
  std::shared_ptr<ov::Model> model;
  ov::CompiledModel compiled_model;
  ov::InferRequest infer_request;

  // @brief pre-processing step for paddle text detection
  static ov::Tensor detect_pre_processing(const cv::Mat &m,
                                          ov::CompiledModel compiled_model,
                                          cv::Size2f &ratio,
                                          const DetectConfig &config);

  // &brief compare 2 rectangle
  static bool compare_result(const cv::Rect &a, const cv::Rect &b,
                             float tolerance = 0);
};

} // namespace segmentation::text::paddle
