#include <execinfo.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/quality/qualityssim.hpp>
#include <sstream>

#include "segmentation/text/paddle/config.hpp"
#include "segmentation/text/paddle/detect.hpp"

namespace paddle = segmentation::text::paddle;

TEST(TestPaddle, TestDetection) {
  std::filesystem::path model_path(
      "../../../../../../models/PaddlePaddle/PP-OCRv5_mobile_det");
  paddle::DetectConfig config = paddle::read_detect_config(model_path, "CPU");
  config.db_config.dilate_kernel = cv::Size(8, 8);
  auto detection = paddle::Detection(config);
  cv::Mat m = cv::imread("data/text1.png");
  auto detected = detection.segment(m);
  paddle::Detection::sort_results(detected);
  ASSERT_EQ(detected.size(), 5);
  std::ostringstream ss;
  int index = 0;
  for (auto d : detected) {
    if (index == 1) {
      index++;
      continue;
    }
    ss.clear();
    ss.str("");
    ss << "data/segment_" << ++index << ".jpeg";
    auto exp = cv::imread(ss.str());
    cv::Scalar ssim_score =
        cv::quality::QualitySSIM::compute(d.segment, exp, cv::noArray());
    ASSERT_GT(ssim_score[0], 0.97);
    ASSERT_GT(ssim_score[1], 0.97);
    ASSERT_GT(ssim_score[2], 0.97);
  }
}
