#include <gtest/gtest.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/quality/qualityssim.hpp>
#include <sstream>
#include <vector>

#include "pipeline/image/resize_op.hpp"
#include "pipeline/image/segment_op.hpp"

namespace img = pipeline::image::op;

TEST(TestSegmenting, TestSameSizeDBSegment) {
  cv::Mat m = cv::imread("image/data/binary.png");
  auto cropper = img::create_cropper(m, img::align_horizontal);
  auto result = img::db_segmentation(m, cropper, {0.3, 1000, 0.6, 1.0}, {1, 1});
  ASSERT_EQ(result.size(), 6);
  std::stringstream ss;
  for (int i = 0; i < result.size(); i++) {
    auto segment = result[i].segment;
    ss.clear();
    ss.str("");
    ss << "image/data/binary_seg_" << i + 1 << ".png";
    auto exp = cv::imread(ss.str());
    ASSERT_EQ(segment.empty(), exp.empty());
    ASSERT_EQ(segment.cols, exp.cols);
    ASSERT_EQ(segment.rows, exp.rows);
    ASSERT_EQ(segment.type(), exp.type());
    cv::Mat diff;
    cv::absdiff(segment, exp, diff);
    ASSERT_EQ(cv::countNonZero(diff.reshape(1)), 0) << "index: " << i + 1;
  }
}

TEST(TestSegmenting, TestScaleSizeDBSegment) {
  cv::Mat m = cv::imread("image/data/binary.png");
  auto cropper = img::create_cropper(m, img::align_horizontal);
  // resize to the closest grid block of 32x32
  cv::Size2f ratio;
  cv::Mat resized = img::resize_n_mutiply(m, {}, cv::Size(32, 32), ratio,
                                          cv::InterpolationFlags::INTER_LINEAR);
  auto result =
      img::db_segmentation(resized, cropper, {0.3, 1000, 0.6, 1.0}, ratio);
  ASSERT_EQ(result.size(), 6);
  std::stringstream ss;
  for (int i = 0; i < result.size(); i++) {
    auto segment = result[i].segment;
    ss.clear();
    ss.str("");
    ss << "image/data/binary_seg_" << i + 1 << ".png";
    auto exp = cv::imread(ss.str());
    ASSERT_EQ(segment.empty(), exp.empty());
    ASSERT_LT(segment.cols - exp.cols, 4);
    ASSERT_LT(segment.rows - exp.rows, 4);
    ASSERT_EQ(segment.type(), exp.type());

    // compare center image
    int min_width = std::min(segment.cols, exp.cols);
    int min_height = std::min(segment.rows, exp.rows);
    int x1 = (segment.cols - min_width) / 2;
    int y1 = (segment.rows - min_height) / 2;
    int x2 = (exp.cols - min_width) / 2;
    int y2 = (exp.rows - min_height) / 2;

    // Define the Region of Interest (ROI) for both images
    cv::Rect roi1(x1, y1, min_width, min_height);
    cv::Rect roi2(x2, y2, min_width, min_height);

    cv::Mat ssim_map;
    cv::Scalar ssim_score =
        cv::quality::QualitySSIM::compute(segment(roi1), exp(roi2), ssim_map);

    ASSERT_GT(ssim_score[0], 0.85);
    ASSERT_GT(ssim_score[1], 0.85);
    ASSERT_GT(ssim_score[2], 0.85);
  }
}
