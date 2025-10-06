#include <gtest/gtest.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/runtime/core.hpp>
#include <vector>

#include "pipeline/image/permute_op.hpp"

namespace img = pipeline::image::op;

TEST(TestPermute, Test2Channel) {
  cv::Mat mch(2, 2, CV_32FC2, {10.0f, 15.0f});
  std::vector<float> vec(2 * 2 * 2);
  std::vector<float> expect = {10, 10, 10, 10, 15, 15, 15, 15};
  img::permute(mch, vec.data());
  ASSERT_EQ(vec, expect);
}

TEST(TestPermute, Test3Channel) {
  cv::Mat mch(2, 2, CV_32FC3, {10.0f, 15.0f, 5.0f});
  std::vector<float> vec(3 * 2 * 2);
  std::vector<float> expect = {10, 10, 10, 10, 15, 15, 15, 15, 5, 5, 5, 5};
  img::permute(mch, vec.data());
  ASSERT_EQ(vec, expect);
}

TEST(TestPermute, Test4Channel) {
  cv::Mat mch(2, 2, CV_32FC4, {10.0f, 15.0f, 5.0f, 1.0f});
  std::vector<float> vec(4 * 2 * 2);
  std::vector<float> expect = {10, 10, 10, 10, 15, 15, 15, 15,
                               5,  5,  5,  5,  1,  1,  1,  1};
  img::permute(mch, vec.data());
  ASSERT_EQ(vec, expect);
}
