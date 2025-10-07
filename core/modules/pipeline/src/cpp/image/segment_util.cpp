#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include "pipeline/image/segment_util.hpp"

namespace pipeline::image::op {

// @brief cropped the image based on the given points
cv::Mat crop(const cv::Mat &src, const std::vector<cv::Point2f> &points,
             const std::optional<CropModifier> modifer) {
  // convert floating number to int
  std::vector<cv::Point> polyint;
  polyint.reserve(points.size());
  for (const cv::Point2f &p : points)
    polyint.emplace_back(cv::Point(cvRound(p.x), cvRound(p.y)));

  cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
  cv::fillPoly(mask, polyint, cv::Scalar(255));
  cv::Mat masked;
  src.copyTo(masked, mask);

  cv::Rect bbox = cv::boundingRect(polyint);
  bbox.x = std::max(0, bbox.x);
  bbox.y = std::max(0, bbox.y);
  bbox.width = std::min(src.cols - bbox.x, bbox.width);
  bbox.height = std::min(src.rows - bbox.y, bbox.height);
  if (bbox.width <= 0 || bbox.height <= 0)
    return cv::Mat();

  cv::Mat cropped = masked(bbox);

  if (modifer.has_value()) {
    auto box = cv::minAreaRect(points);
    box.center.x -= bbox.x;
    box.center.y -= bbox.y;
    auto fn = modifer.value();
    cropped = fn(cropped, points, box);
  }

  // remove any unecessary surrounding the polygon target content
  cv::Mat gray;
  if (cropped.channels() == 3) {
    cv::cvtColor(cropped, gray, cv::COLOR_BGR2GRAY);
  } else if (cropped.channels() == 4) {
    cv::cvtColor(cropped, gray, cv::COLOR_BGRA2GRAY);
  } else {
    gray = cropped.clone();
  }

  // threshold/boolean mask (single-channel CV_8UC1)
  cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY);

  if (cv::countNonZero(gray) == 0) {
    return cropped;
  }

  cv::Rect tight = cv::boundingRect(gray);
  // guard bounds just in case
  tight.x = std::max(0, tight.x);
  tight.y = std::max(0, tight.y);
  tight &= cv::Rect(0, 0, cropped.cols, cropped.rows);
  return cropped(tight).clone();
}

// @brief create cropper function
Cropper create_cropper(const cv::Mat &m,
                       const std::optional<CropModifier> modifer) {
  return [m, modifer](const std::vector<cv::Point2f> points) -> cv::Mat {
    return crop(m, points, modifer);
  };
}

// @brief rotate the src to align horizontally if the rotated.angle is not 0 or
// 90 degree.
cv::Mat align_horizontal(cv::Mat &src, const std::vector<cv::Point2f> &points,
                         const cv::RotatedRect &rotated) {
  float angle = rotated.angle;
  cv::Size2f box_size = rotated.size;
  if (box_size.width < box_size.height) {
    angle += 90.0f;
    std::swap(box_size.width, box_size.height);
  }
  cv::Mat m;
  cv::Mat rot_mat = cv::getRotationMatrix2D(rotated.center, angle, 1.0);
  cv::warpAffine(src, m, rot_mat, src.size(), cv::INTER_CUBIC);
  return m;
}

} // namespace pipeline::image::op
