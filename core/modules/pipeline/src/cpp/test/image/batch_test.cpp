#include <algorithm>
#include <gtest/gtest.h>
#include <memory>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/runtime/core.hpp>
#include <vector>

#include "pipeline/batch.hpp"
#include "pipeline/image/permute_op.hpp"

struct BatchTestCaseParams {
  std::vector<cv::Rect> rects;
  std::vector<cv::Rect> outputs;
  int process_call_count;
  cv::Size max_sizes;
};

class BatchProcessing : public pipeline::IProcessing<cv::Size, cv::Rect, int> {
public:
  cv::Size max_sizes;
  int process_count;
  std::vector<cv::Rect> segments;

  BatchProcessing(cv::Size max_sizes, int process_count,
                  std::vector<cv::Rect> segments = {})
      : max_sizes(max_sizes), process_count(process_count), segments(segments) {
  }

  BatchProcessing(BatchProcessing &&other) {}

  BatchProcessing &operator=(BatchProcessing &&other) { return *this; }

  cv::Size batch_initialize(int start, int count,
                            const std::vector<cv::Rect> &segments) override {
    this->segments.clear();
    this->max_sizes.width = 0;
    this->max_sizes.height = 0;
    for (auto s : segments) {
      this->max_sizes.width = std::max(this->max_sizes.width, s.width);
      this->max_sizes.height = std::max(this->max_sizes.height, s.height);
    }
    return this->max_sizes;
  }

  // @brief handling each processing item
  cv::Rect batch_processing(int index, const cv::Mat &origin, cv::Size &t,
                            cv::Rect &segment) override {
    this->process_count = index + 1;
    this->segments.push_back(segment);
    return segment;
  }

  // @brief call after all items has been processed.
  int batch_post_processing(cv::Size &t, int start, int count,
                            const std::vector<cv::Rect> &segments) override {
    return t.area();
  }
};

class BatchTestCase : public ::testing::TestWithParam<BatchTestCaseParams> {
protected:
  // This static method runs once before any test in this suite
  static void SetUpTestSuite() {}

  // This static method runs once after all tests in this suite
  static void TearDownTestSuite() {}
};

INSTANTIATE_TEST_SUITE_P(BatchTestSuite, // A unique name for the test suite
                         BatchTestCase,  // The test fixture class
                         ::testing::Values(BatchTestCaseParams{
                             {
                                 cv::Rect(1, 1, 5, 5),
                                 cv::Rect(7, 1, 5, 5),
                                 cv::Rect(1, 7, 5, 5),
                                 cv::Rect(7, 7, 5, 5),
                             },
                             {
                                 cv::Rect(1, 1, 5, 5),
                                 cv::Rect(7, 1, 5, 5),
                                 cv::Rect(1, 7, 5, 5),
                                 cv::Rect(7, 7, 5, 5),
                             },
                             4,
                             cv::Size(5, 5),
                         }));

TEST_P(BatchTestCase, Batch) {
  const BatchTestCaseParams &params = GetParam();
  auto batch_processing = std::make_unique<BatchProcessing>(cv::Size(0, 0), 0);
  auto bp_ptr = batch_processing.get();
  auto toIface =
      static_cast<std::unique_ptr<pipeline::IProcessing<cv::Size, cv::Rect, int>>>(
          std::move(batch_processing));
  cv::Mat m;
  pipeline::Source<cv::Rect> source(m, params.rects);
  pipeline::BatchProcessor processor(source, 6, std::move(toIface));
  for (const auto &batch : processor) {
    ASSERT_EQ(bp_ptr->segments, params.outputs);
    ASSERT_EQ(bp_ptr->max_sizes, params.max_sizes);
    ASSERT_EQ(bp_ptr->process_count, params.process_call_count);
    ASSERT_EQ(batch, params.max_sizes.area());
  }
}
