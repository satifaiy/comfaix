#include <gtest/gtest.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/runtime/core.hpp>

#include "pipeline/image/resize_op.hpp"

namespace img = pipeline::image::op;

TEST(TestResize, TestNMultipleResizing) {
  cv::Mat m(32, 40, CV_8UC1);
  cv::Size2f ratio;
  auto resize =
      img::resize_n_mutiply(m, cv::Size(120, 120), cv::Size(12, 12), ratio,
                            cv::InterpolationFlags::INTER_LINEAR);
  ASSERT_EQ(36, resize.cols);
  ASSERT_EQ(36, resize.rows);
  ASSERT_FLOAT_EQ(0.9f, ratio.width);
  ASSERT_FLOAT_EQ(1.125f, ratio.height);

  m = cv::Mat(32, 42, CV_8UC1);
  resize = img::resize_n_mutiply(m, cv::Size(120, 120), cv::Size(12, 12), ratio,
                                 cv::InterpolationFlags::INTER_LINEAR);
  ASSERT_EQ(48, resize.cols);
  ASSERT_EQ(36, resize.rows);
  ASSERT_FLOAT_EQ(1.142857143f, ratio.width);
  ASSERT_FLOAT_EQ(1.125f, ratio.height);

  m = cv::Mat(124, 82, CV_8UC1);
  resize = img::resize_n_mutiply(m, cv::Size(120, 120), cv::Size(12, 12), ratio,
                                 cv::InterpolationFlags::INTER_LINEAR);
  ASSERT_EQ(84, resize.cols);
  ASSERT_EQ(120, resize.rows);
  ASSERT_FLOAT_EQ(1.024390244f, ratio.width);
  ASSERT_FLOAT_EQ(0.967741935f, ratio.height);

  m = cv::Mat(124, 82, CV_8UC1);
  resize = img::resize_n_mutiply(m, cv::Size(198, 140), cv::Size(22, 14), ratio,
                                 cv::InterpolationFlags::INTER_LINEAR);
  ASSERT_EQ(88, resize.cols);
  ASSERT_EQ(126, resize.rows);
  ASSERT_FLOAT_EQ(1.073170732f, ratio.width);
  ASSERT_FLOAT_EQ(1.016129032f, ratio.height);

  m = cv::Mat(150, 210, CV_8UC1);
  resize = img::resize_n_mutiply(m, cv::Size(198, 140), cv::Size(22, 14), ratio,
                                 cv::InterpolationFlags::INTER_LINEAR);
  ASSERT_EQ(198, resize.cols);
  ASSERT_EQ(140, resize.rows);
  ASSERT_FLOAT_EQ(0.942857143f, ratio.width);
  ASSERT_FLOAT_EQ(0.933333333f, ratio.height);
}
