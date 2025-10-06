#pragma one

#include <algorithm>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

#include "common/ovs_assert.hpp"

namespace pipeline::image::op {

// @brief resize image to fit width and height and does not keep aspect ration
inline cv::Mat resize(cv::Mat &m, int width, int height,
                      cv::InterpolationFlags flags) {
  cv::Mat resized;
  cv::resize(m, resized, cv::Size(width, height), 0.f, 0.f, flags);
  return resized;
}

inline cv::Mat resize_fit_width(cv::Mat &m, int width,
                                cv::InterpolationFlags flags) {
  auto ratio = 1.0f * m.cols / m.rows;
  cv::Mat resized;
  cv::resize(m, resized, cv::Size(width, width / ratio), 0.f, 0.f, flags);
  return resized;
}

inline cv::Mat resize_fit_height(cv::Mat &m, int height,
                                 cv::InterpolationFlags flags) {
  auto ratio = 1.0f * m.cols / m.rows;
  cv::Mat resized;
  cv::resize(m, resized, cv::Size(ratio * height, height), 0.f, 0.f, flags);
  return resized;
}

// @brief resize the image to fit grid of multiple block.
inline cv::Mat resize_n_mutiply(cv::Mat &m, cv::Size max, cv::Size block,
                                cv::Size2f &ratio,
                                cv::InterpolationFlags flags) {
  OVS_ASSERT(!block.empty(), "block grid must not be empty");
  OVS_ASSERT(max.width == 0 || max.width % block.width == 0, "maximum width ",
             max.width, " is not multiple of ", block.width);
  OVS_ASSERT(max.height == 0 || max.height % block.height == 0,
             "maximum height ", max.height, " is not multiple of ",
             block.height);
  cv::Mat resized;
  cv::Size size(m.cols, m.rows);
  if (size.width > size.height) {
    if (size.width > max.width) {
      // limit maximum width
      size.height = (size.height * max.width) / size.width;
      size.width = max.width;
      if (size.height > max.height) {
        // force height to fit with maximum
        size.height = max.height;
      }
    }
  } else {
    if (size.height > max.height) {
      // limit maximum width
      size.width = (size.width * max.height) / size.height;
      size.height = max.height;
      if (size.width > max.width) {
        // force width to fit with maximum
        size.width = max.width;
      }
    }
  }
  // aligment to grid block
  size.width = std::max(
      int(round(float(size.width) / block.width) * block.width), block.width);
  size.height =
      std::max(int(round(float(size.height) / block.height) * block.height),
               block.height);
  cv::resize(m, resized, size);
  ratio.width = float(size.width) / m.cols;
  ratio.height = float(size.height) / m.rows;
  return resized;
}

} // namespace pipeline::image::op
