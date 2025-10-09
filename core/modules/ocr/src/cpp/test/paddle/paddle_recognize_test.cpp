#include <execinfo.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <vector>

#include "ocr/paddle/recognize.hpp"
#include "segmentation/text/paddle/config.hpp"
#include "segmentation/text/paddle/detect.hpp"

namespace seg_paddle = segmentation::text::paddle;
namespace reg_paddle = ocr::paddle;

TEST(TestPaddle, TestDetection) {
  std::filesystem::path det_path(
      "../../../../../../models/PaddlePaddle/PP-OCRv5_mobile_det");
  std::filesystem::path reg_path(
      "../../../../../../models/PaddlePaddle/en_PP-OCRv5_mobile_rec");

  seg_paddle::DetectConfig config =
      seg_paddle::read_detect_config(det_path, "CPU");
  config.db_config.dilate_kernel = cv::Size(8, 8);
  auto seg = std::make_unique<seg_paddle::Detection>(config);
  auto recognize = reg_paddle::Recognize(reg_path, std::move(seg), "CPU");

  cv::Mat m = cv::imread("data/text.png");
  auto results = recognize.recognize(m);
  std::vector<std::string> expected = {
      "Semantic segmentation enhances image search by",
      "enabling precise, pixel-level understanding of image",
      "content. Unlike traditional methods that rely on tags or",
      "object detection boxes, semantic segmentation classifies",
  };

  ASSERT_EQ(results.size(), expected.size());
  for (int i = 0; i < results.size(); i++) {
    ASSERT_EQ(results[i].segment, expected[i]);
  }
}
