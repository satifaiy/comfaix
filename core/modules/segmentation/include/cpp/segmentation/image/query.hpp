#pragma once

#include <opencv2/core/mat.hpp>
#include <vector>

namespace segmentation::image {

// @brief query data
template <typename T, typename S> struct Query {
  // data use to comparison depend on mechanism
  T data;

  // attribute, an addition data to processing
  S attribute;

  // minimum score requirement
  double min_score = 0.90;
};

// @brief build queries from image
template <typename T, typename S>
std::vector<Query<T, S>> build_query(const std::vector<cv::Mat> &queries);

} // namespace segmentation::image
