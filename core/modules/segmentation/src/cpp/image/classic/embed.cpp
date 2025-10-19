#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/quality/qualityssim.hpp>

#include "common/ovs_assert.hpp"
#include "segmentation/image/classic/embed.hpp"

namespace segmentation::image::classic {

// @brief create simple Similarity using opencv qualitiy measurement
Similarity cv_quality() {
  return [](const cv::Mat &a, const cv::Mat &b) -> double {
    OVS_ASSERT(a.channels() == b.channels(),
               "opencv qualityssim require 2 input to have the same channels");
    auto ia = a;
    auto ib = b;
    if (a.rows != b.rows || a.cols != b.cols) {
      cv::Size size((a.cols + b.cols) / 2, (a.rows + b.rows) / 2);
      cv::resize(a, ia, size);
      cv::resize(b, ib, size);
    }
    auto scalar = cv::quality::QualitySSIM::compute(ia, ib, cv::noArray());
    double total = 0;
    switch (a.channels()) {
    case 4:
      total += scalar[3];
    case 3:
      total += scalar[2];
    case 2:
      total += scalar[1];
    case 1:
      total += scalar[0];
      break;
    default:
      return 0;
    }
    return total / a.channels();
  };
}

} // namespace segmentation::image::classic
