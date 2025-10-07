#pragma once

#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "common/segmentation.hpp"
#include "pipeline/image/segment_util.hpp"

namespace common = comfaix::common;

namespace pipeline::image::op {

// @brief define algorithm use for segmentation
enum class Algorithm {
  PSENet, // Progressive Scale Expansion Network
  DBNet,  // Differentiable Binarization
  MaskRCNN,
};

// @brief configuration for segmentation of Differentiable Binarization
struct DBNetConfig {
  double thresh;
  int max_candidates;
  double box_thresh;
  double unclip_ratio;
};


// @brief segmentation using Differentiable Binarization
// @param bitmap must be an BGR color order or a grayscale.
std::vector<common::Result<cv::Mat, std::vector<cv::Point2f>>>
db_segmentation(cv::Mat &bitmap, Cropper cropper, const DBNetConfig &config,
                const cv::Size2f ratio, const cv::Mat &in_predict = {});

// @brief transform points by unclip
void unclip(const std::vector<cv::Point2f> &inPoly,
            std::vector<cv::Point2f> &outPoly, const double unclipRatio);

} // namespace pipeline::image::op
