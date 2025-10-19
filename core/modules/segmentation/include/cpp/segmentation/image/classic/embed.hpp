#include <functional>
#include <opencv2/core/mat.hpp>

namespace segmentation::image::classic {

// @brief function provide similarity check
using Similarity = std::function<double(const cv::Mat &a, const cv::Mat &b)>;

// @brief create simple Similarity using opencv qualitiy measurement
Similarity cv_quality();

} // namespace segmentation::image::classic
