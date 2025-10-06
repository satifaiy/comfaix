#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include "pipeline/image/score.hpp"

namespace pipeline::image {

// @brief calculate segmentation score using contour.
float contour_score(cv::Mat &predict,
                       const std::vector<cv::Point> &contour) {
  cv::Rect rect = cv::boundingRect(contour);
  int xmin = std::max(rect.x, 0);
  int xmax = std::min(rect.x + rect.width, predict.cols - 1);
  int ymin = std::max(rect.y, 0);
  int ymax = std::min(rect.y + rect.height, predict.rows - 1);

  cv::Mat binROI =
      predict(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));

  cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8U);
  std::vector<cv::Point> roiContour;
  for (size_t i = 0; i < contour.size(); i++) {
    cv::Point pt = cv::Point(contour[i].x - xmin, contour[i].y - ymin);
    roiContour.push_back(pt);
  }
  std::vector<std::vector<cv::Point>> roiContours = {roiContour};
  fillPoly(mask, roiContours, cv::Scalar(1));
  double score = cv::mean(binROI, mask).val[0];
  return float(score);
}

} // namespace pipeline::image
