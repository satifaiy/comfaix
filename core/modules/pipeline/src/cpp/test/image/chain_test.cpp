#include <gtest/gtest.h>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/tensor.hpp>
#include <vector>

#include "pipeline/batch.hpp"
#include "pipeline/image/normalize_op.hpp"
#include "pipeline/image/padding_op.hpp"
#include "pipeline/image/permute_op.hpp"
#include "pipeline/image/resize_op.hpp"

namespace img = pipeline::image::op;

bool is_mats_identical(const cv::Mat &m1, const cv::Mat &m2) {
  if (m1.empty() != m2.empty()) {
    return false;
  }
  if (m1.cols != m2.cols || m1.rows != m2.rows || m1.type() != m2.type()) {
    return false;
  }
  cv::Mat diff;
  cv::compare(m1, m2, diff, cv::CMP_EQ);
  return cv::countNonZero(diff.reshape(1)) == 0;
}

struct Segment {
  cv::Rect rect;
  cv::Mat chunk;
};

struct BatchData {
  std::vector<Segment> segments;
  cv::Size max_size;
  int total;
  std::vector<float> input;
  ov::Tensor tensor;
};

class ChainProcessing
    : public pipeline::IProcessing<BatchData, Segment, ov::Tensor> {
public:
  BatchData batch_data;

  ChainProcessing() {}

  BatchData batch_initialize(int start, int count,
                             const std::vector<Segment> &segments) override {
    // clear previous data
    this->batch_data.total = count;
    this->batch_data.segments.clear();
    this->batch_data.max_size.width = 0;
    this->batch_data.max_size.height = 12;
    this->batch_data.input.clear();

    // get max width & height
    float nw = 0;
    for (int i = 0; i < count; i++) {
      auto nw = (12.0f * segments[start + i].rect.width) /
                segments[start + i].rect.height;
      this->batch_data.max_size.width =
          std::max(this->batch_data.max_size.width, int(nw));
    }

    this->batch_data.input.resize(count * 3 * this->batch_data.max_size.height *
                                  this->batch_data.max_size.width);
    return this->batch_data;
  }

  // @brief handling each processing item
  Segment batch_processing(int index, const cv::Mat &origin, BatchData &t,
                           Segment &segment) override {
    cv::Mat seg = origin(segment.rect);
    seg = img::resize_fit_height(seg, this->batch_data.max_size.height,
                                 cv::InterpolationFlags::INTER_LINEAR);
    seg = img::padding_right(seg, this->batch_data.max_size.width - seg.cols,
                             cv::BorderTypes::BORDER_CONSTANT, {51, 102, 153});
    seg = img::min_max_normalize(seg, CV_32FC3);
    seg = img::standard_score_normalize(seg, CV_32FC1, 0.5, 0.5);
    float *data = (float *)this->batch_data.input.data();
    data += index * 3 * this->batch_data.max_size.width *
            this->batch_data.max_size.height;
    img::permute(seg, data);
    segment.chunk = seg;
    t.segments.push_back(segment);
    return segment;
  }

  // @brief call after all items has been processed.
  ov::Tensor
  batch_post_processing(BatchData &t, int start, int count,
                        const std::vector<Segment> &segments) override {
    ov::Shape shape = {size_t(count), 3, size_t(t.max_size.height),
                       size_t(t.max_size.width)};
    ov::Tensor tensor(ov::element::f32, shape, this->batch_data.input.data());
    return tensor;
  }
};

struct ProprocessTestCaseParams {
  cv::Mat origin;
  std::vector<Segment> segments;
  std::vector<Segment> outputs;
  std::vector<cv::Size> max_sizes;
  std::vector<int> totals;
  std::vector<std::vector<float>> nchw;
};

class PreProcessTestCase
    : public ::testing::TestWithParam<ProprocessTestCaseParams> {
protected:
  // This static method runs once before any test in this suite
  static void SetUpTestSuite() {}

  // This static method runs once after all tests in this suite
  static void TearDownTestSuite() {}
};

std::vector<float> flatvector(int count, int total, std::vector<float> val) {
  int chunk_size = total * val.size();
  std::vector<float> flatten(count * chunk_size);
  for (int c = 0; c < count; c++) {
    for (int i = 0; i < total; i++) {
      for (int j = 0; j < val.size(); j++) {
        flatten[(j * total) + i + (c * chunk_size)] = val[j];
      }
    }
  }
  return flatten;
}

INSTANTIATE_TEST_SUITE_P(
    PreprocessTestSuite, // A unique name for the test suite
    PreProcessTestCase,  // The test fixture class
    ::testing::Values(ProprocessTestCaseParams{
        cv::Mat(50, 50, CV_8UC3, {51, 102, 153}),
        {
            {cv::Rect(1, 1, 5, 5)},
            {cv::Rect(7, 1, 10, 5)},
            {cv::Rect(1, 7, 5, 10)},
            {cv::Rect(7, 7, 15, 25)},
        },
        {
            {cv::Rect(1, 1, 5, 5),
             cv::Mat(12, 24, CV_32FC3, {-0.6, -0.2, 0.2})},
            {cv::Rect(7, 1, 10, 5),
             cv::Mat(12, 24, CV_32FC3, {-0.6, -0.2, 0.2})},
            {cv::Rect(1, 7, 5, 10),
             cv::Mat(12, 7, CV_32FC3, {-0.6, -0.2, 0.2})},
            {cv::Rect(7, 7, 15, 25),
             cv::Mat(12, 7, CV_32FC3, {-0.6, -0.2, 0.2})},
        },
        {cv::Size(24, 12), cv::Size(7, 12)},
        {2, 2},
        {
            flatvector(2, 12 * 24, {-0.6, -0.2, 0.2}),
            flatvector(2, 12 * 7, {-0.6, -0.2, 0.2}),
        },
    }));

TEST_P(PreProcessTestCase, Preprocessing) {
  const ProprocessTestCaseParams &params = GetParam();
  auto chain_processing = std::make_unique<ChainProcessing>();
  auto cp_ptr = chain_processing.get();
  auto toIface = static_cast<
      std::unique_ptr<pipeline::IProcessing<BatchData, Segment, ov::Tensor>>>(
      std::move(chain_processing));

  pipeline::Source<Segment> source(params.origin, params.segments);
  pipeline::BatchProcessor processor(source, 2, std::move(toIface));
  int batch_count = 0;
  for (auto &tensor : processor) {
    ov::Shape shape = {size_t(params.totals[batch_count]), 3,
                       size_t(params.max_sizes[batch_count].height),
                       size_t(params.max_sizes[batch_count].width)};
    ASSERT_EQ(ov::element::f32, tensor.get_element_type());
    ASSERT_EQ(shape, tensor.get_shape());

    auto vec = params.nchw[batch_count];
    auto ptrf = vec.data();
    float *tensor_data = tensor.data<float>();
    for (int i = 0; i < vec.size(); i++) {
      ASSERT_NEAR(ptrf[i], tensor_data[i], 1e-6f) << " index: " << i;
    }

    auto segments = cp_ptr->batch_data.segments;
    for (int i = 0; i < segments.size(); i++) {
      ASSERT_EQ(segments[i].rect, params.outputs[i + (2 * batch_count)].rect);
      ASSERT_TRUE(is_mats_identical(
          segments[i].chunk, params.outputs[i + (2 * batch_count)].chunk));
    }
    ASSERT_EQ(cp_ptr->batch_data.max_size, params.max_sizes[batch_count]);
    ASSERT_EQ(cp_ptr->batch_data.total, params.totals[batch_count]);
    ASSERT_EQ(cp_ptr->batch_data.input.size(), params.nchw[batch_count].size());
    for (int i = 0; i < params.nchw[batch_count].size(); i++) {
      ASSERT_NEAR(cp_ptr->batch_data.input[i], params.nchw[batch_count][i],
                  1e-6f);
    }
    batch_count++;
  }
}
