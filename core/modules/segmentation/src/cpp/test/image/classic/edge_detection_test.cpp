#include <gtest/gtest.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/quality/qualityssim.hpp>
#include <vector>

#include "segmentation/image/classic/detection.hpp"
#include "segmentation/image/classic/edge.hpp"
#include "segmentation/image/classic/embed.hpp"
#include "segmentation/image/classic/query.hpp"

namespace classic = segmentation::image::classic;

TEST(TestClassic, TestDetectionSimpleUIWithEdge) {
  auto search = cv::imread("data/image/classic/ui_test.jpeg");
  auto edge = classic::sobel(3, 3);
  classic::Detection detection(edge, classic::cv_quality());
  std::vector<cv::Mat> all_queries = {
      cv::imread("data/image/classic/ui_button1.png"),  // 2
      cv::imread("data/image/classic/ui_button2.png"),  // 2
      cv::imread("data/image/classic/ui_button3.png"),  // 1
      cv::imread("data/image/classic/ui_button4.png"),  // 1
      cv::imread("data/image/classic/ui_checkbox.png"), // 1
      cv::imread("data/image/classic/ui_rotary.png"),   // 1
      cv::imread("data/image/classic/ui_star1.png"),    // 3
      cv::imread("data/image/classic/ui_star2.png"),    // 1
  };
  const double min_score = 0.60;
  auto queries = detection.build_queries(all_queries, {min_score});
  auto results = detection.detect(search, queries);
  ASSERT_EQ(results.size(), 12);

  for (const auto r : results) {
    cv::Mat target = r.segment.segment;
    cv::Mat query = queries.data[r.segment.index].data;
    if (target.rows != query.rows || target.cols != query.cols) {
      target = target.clone();
      cv::resize(target, target, query.size());
    }
    auto scalar =
        cv::quality::QualitySSIM::compute(target, query, cv::noArray());
    auto sum = (scalar[0] + scalar[1] + scalar[2] + scalar[3]);
    ASSERT_GT(sum / query.channels(), min_score);

    cv::Mat origin;
    ASSERT_TRUE(classic::crop(search, origin, r.roi));
    ASSERT_EQ(query.channels(), origin.channels());
    if (origin.rows != query.rows || origin.cols != query.cols) {
      cv::resize(origin, origin, query.size());
    }
    scalar = cv::quality::QualitySSIM::compute(origin, query, cv::noArray());
    sum = (scalar[0] + scalar[1] + scalar[2] + scalar[3]);
    ASSERT_GT(sum / query.channels(), min_score);
  }
}
