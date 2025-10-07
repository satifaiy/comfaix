#pragma once

#include <opencv2/core/mat.hpp>

namespace pipeline::image {

// @brief calculate segmentation score using contour
float contour_score(cv::Mat &predict, const std::vector<cv::Point>& contour);

} // namespace pipeline::image
