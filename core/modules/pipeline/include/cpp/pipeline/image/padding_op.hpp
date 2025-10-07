#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

namespace pipeline::image::op {

// @brief extend image to left, top, right or bottom to fit desire with or
// height
inline cv::Mat padding(cv::Mat &m, int left, int top, int right, int bottom,
                       cv::BorderTypes type, const cv::Scalar color = {}) {
  cv::copyMakeBorder(m, m, top, bottom, left, right, type, color);
  return m;
}

inline cv::Mat padding_left(cv::Mat &m, int left, cv::BorderTypes type,
                            const cv::Scalar color = {}) {
  cv::copyMakeBorder(m, m, 0, 0, left, 0, type, color);
  return m;
}

inline cv::Mat padding_top(cv::Mat &m, int top, cv::BorderTypes type,
                           const cv::Scalar color = {}) {
  cv::copyMakeBorder(m, m, top, 0, 0, 0, type, color);
  return m;
}

inline cv::Mat padding_right(cv::Mat &m, int right, cv::BorderTypes type,
                             const cv::Scalar color = {}) {
  cv::copyMakeBorder(m, m, 0, 0, 0, right, type, color);
  return m;
}

inline cv::Mat padding_bottom(cv::Mat &m, int bottom, cv::BorderTypes type,
                              const cv::Scalar color = {}) {
  cv::copyMakeBorder(m, m, 0, bottom, 0, 0, type, color);
  return m;
}

} // namespace pipeline::image::op
