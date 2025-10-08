#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "common/ovs_assert.hpp"
#include "pipeline/image/score.hpp"
#include "pipeline/image/segment_op.hpp"

namespace pipeline::image::op {

// @brief segmentation using Differentiable Binarization
std::vector<common::Result<cv::Mat, std::vector<cv::Point2f>>>
db_segmentation(cv::Mat &bitmap, Cropper cropper, const DBNetConfig &config,
                const cv::Size size, const cv::Size2f ratio,
                const cv::Mat &in_predict) {
  // if no prediction given then fallback to bitmap
  cv::Mat predict = in_predict;
  if (in_predict.empty()) {
    predict = bitmap;
  }

  if (bitmap.type() != CV_8UC1) {
    cv::Mat single_channel;
    cv::cvtColor(bitmap, single_channel, cv::COLOR_BGR2GRAY);
    OVS_ASSERT(single_channel.type() == CV_8UC1, "expect type ", CV_8UC1,
               " but got ", single_channel.type());
    bitmap = single_channel;
  }
  cv::Mat threshold;
  cv::threshold(bitmap, threshold, config.thresh * 255, 255, cv::THRESH_BINARY);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(threshold, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

  size_t num_candidate = std::min(
      contours.size(),
      (size_t)(config.max_candidates > 0 ? config.max_candidates : INT_MAX));

  std::vector<common::Result<cv::Mat, std::vector<cv::Point2f>>> results;

  for (size_t i = 0; i < num_candidate; i++) {
    std::vector<cv::Point> &contour = contours[i];

    float score = contour_score(predict, contour);
    if (score < config.box_thresh)
      continue;

    std::vector<cv::Point> contourScaled;
    contourScaled.reserve(contour.size());
    for (size_t j = 0; j < contour.size(); j++) {
      contourScaled.push_back(cv::Point(int(contour[j].x), int(contour[j].y)));
    }

    // Unclip
    cv::RotatedRect box = cv::minAreaRect(contourScaled);
    float minLen =
        std::min(box.size.height / ratio.width, box.size.width / ratio.height);

    // Filter very small boxes
    if (minLen < 3)
      continue;

    const float angle_threshold =
        60; // do not expect vertical text, TODO detection algo property
    bool swap_size = false;
    if (box.size.width <
        box.size.height) // horizontal-wide text area is expected
      swap_size = true;
    else if (std::fabs(box.angle) >=
             angle_threshold) // don't work with vertical rectangles
      swap_size = true;

    if (swap_size) {
      std::swap(box.size.width, box.size.height);
      if (box.angle < 0)
        box.angle += 90;
      else if (box.angle > 0)
        box.angle -= 90;
    }

    cv::Point2f vertex[4];
    box.points(vertex); // order: bl, tl, tr, br
    std::vector<cv::Point2f> approx;
    for (int j = 0; j < 4; j++)
      approx.emplace_back(vertex[j]);
    std::vector<cv::Point2f> polygon;
    unclip(approx, polygon, config.unclip_ratio);
    if (polygon.empty())
      continue;

    // adjust unclip path or rectange base on ratio and input size
    for (int m = 0; m < polygon.size(); m++) {
      polygon[m].x /= ratio.width;
      polygon[m].y /= ratio.height;

      polygon[m].x = std::min(std::max(polygon[m].x, 0.0f), size.width - 1.0f);
      polygon[m].y = std::min(std::max(polygon[m].y, 0.0f), size.height - 1.0f);
    }

    int rect_width, rect_height;
    rect_width = int(sqrt(pow(polygon[0].x - polygon[1].x, 2) +
                          pow(polygon[0].y - polygon[1].y, 2)));
    rect_height = int(sqrt(pow(polygon[0].x - polygon[3].x, 2) +
                           pow(polygon[0].y - polygon[3].y, 2)));
    // skip final rectangle is less than 4
    if (rect_width <= 4 || rect_height <= 4)
      continue;

    cv::Mat cropped = cropper(polygon);
    results.push_back(common::Result{cropped, polygon, score});
  }
  return results;
}

// @brief unclip borrowing the same code from opencv
// TextDetectionModel_DB::unclip
void unclip(const std::vector<cv::Point2f> &inPoly,
            std::vector<cv::Point2f> &outPoly, const double unclip_ratio) {
  double area = cv::contourArea(inPoly);
  double length = cv::arcLength(inPoly, true);

  if (length == 0.)
    return;

  double distance = area * unclip_ratio / length;

  size_t numPoints = inPoly.size();
  std::vector<std::vector<cv::Point2f>> newLines;
  for (size_t i = 0; i < numPoints; i++) {
    std::vector<cv::Point2f> newLine;
    cv::Point pt1 = inPoly[i];
    cv::Point pt2 = inPoly[(i - 1) % numPoints];
    cv::Point vec = pt1 - pt2;
    float unclipDis = (float)(distance / norm(vec));
    cv::Point2f rotateVec = cv::Point2f(vec.y * unclipDis, -vec.x * unclipDis);
    newLine.push_back(cv::Point2f(pt1.x + rotateVec.x, pt1.y + rotateVec.y));
    newLine.push_back(cv::Point2f(pt2.x + rotateVec.x, pt2.y + rotateVec.y));
    newLines.push_back(newLine);
  }

  size_t numLines = newLines.size();
  for (size_t i = 0; i < numLines; i++) {
    cv::Point2f a = newLines[i][0];
    cv::Point2f b = newLines[i][1];
    cv::Point2f c = newLines[(i + 1) % numLines][0];
    cv::Point2f d = newLines[(i + 1) % numLines][1];
    cv::Point2f pt;
    cv::Point2f v1 = b - a;
    cv::Point2f v2 = d - c;
    double cosAngle = (v1.x * v2.x + v1.y * v2.y) / (norm(v1) * norm(v2));

    if (fabs(cosAngle) > 0.7) {
      pt.x = (b.x + c.x) * 0.5;
      pt.y = (b.y + c.y) * 0.5;
    } else {
      double denom = a.x * (double)(d.y - c.y) + b.x * (double)(c.y - d.y) +
                     d.x * (double)(b.y - a.y) + c.x * (double)(a.y - b.y);
      double num = a.x * (double)(d.y - c.y) + c.x * (double)(a.y - d.y) +
                   d.x * (double)(c.y - a.y);
      double s = num / denom;

      pt.x = a.x + s * (b.x - a.x);
      pt.y = a.y + s * (b.y - a.y);
    }
    outPoly.push_back(pt);
  }
}

} // namespace pipeline::image::op
