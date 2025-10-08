#pragma once

#include <opencv2/core/types.hpp>
#include <openvino/core/layout.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/infer_request.hpp>

#include "pipeline/image/segment_op.hpp"

namespace segmentation::text::paddle {

// @brief a configuration to use for detecting text region using paddle detect
// models.
struct DetectConfig {
  std::filesystem::path model_path;
  cv::Scalar means;
  cv::Scalar scales;
  ov::Layout layout;
  cv::Size max_width_height;
  cv::Size block_size;
  bool need_min_max_normalize;
  pipeline::image::op::DBNetConfig db_config;
  std::string device;
};

// @brief support for reading configuration of model version 5
inline namespace v5 {
// @breif read the detection configuration from the model
DetectConfig read_detect_config(std::string path, std::string device);
} // namespace v5

} // namespace segmentation::text::paddle
