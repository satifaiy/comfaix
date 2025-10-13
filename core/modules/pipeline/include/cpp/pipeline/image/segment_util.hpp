#pragma once

#include <functional>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <optional>
#include <vector>

namespace pipeline::image::op {

// @brief providing a modifer to the cropped image. It's usefull for action
// such as rotate the cropped image to align horizontal or vertical.
using CropModifier =
    std::function<cv::Mat(cv::Mat &src, const std::vector<cv::Point2f> &points,
                          const cv::RotatedRect &rotate)>;

// @brief providing a function to crop an image based on a polygon
using Cropper = std::function<cv::Mat(const std::vector<cv::Point2f>)>;

// @brief cropped the image based on the given points
cv::Mat crop(const cv::Mat &src, const std::vector<cv::Point2f> &points,
             const std::optional<CropModifier> modifer = std::nullopt);

// @brief create cropper function
Cropper
create_cropper(const cv::Mat &m,
               const std::optional<CropModifier> modifer = std::nullopt);

// @brief rotate the src to align horizontally if the rotated.angle is not 0 or
// 90 degree.
cv::Mat align_horizontal(cv::Mat &src, const std::vector<cv::Point2f> &points,
                         const cv::RotatedRect &rotated);

} // namespace pipeline::image::op
