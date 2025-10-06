#include <opencv2/core/types.hpp>
#pragma one

#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "common/ovs_assert.hpp"

namespace pipeline::image::op {

inline cv::Mat min_max_normalize(cv::Mat &m, int type, float min_value = 0,
                                 float max_value = 1) {
  // convertTo formula is  Y = α*X + β
  // min max range [a, b] is Y = a + ((X - min(X))*(b-a))/(max(X)-min(X))
  // min(X) = 0 and max(X) = 255 for color range [0,255]
  // min max range Y = a + (X*(b-a))/255
  // CV formula is Y = (X*(b-a))/255 + a
  // ***************************************************
  m.convertTo(m, type, (max_value - min_value) / 255, min_value);
  return m;
}

inline cv::Mat standard_score_normalize(cv::Mat &m, int type, float mean = 0,
                                        float scale = 1) {
  // convertTo formula is  Y = α*X + β
  // standard score is     Y = (X - μ)/σ
  // CV formula is         Y = (x/σ) + (-μ/σ)
  // ***************************************************
  m.convertTo(m, type, 1 / scale, -mean / scale);
  return m;
}

inline cv::Mat standard_score_normalize(cv::Mat &m, int type,
                                        std::vector<float> means,
                                        std::vector<float> scales) {
  OVS_ASSERT(means.size() == m.channels(),
             "image channel does not match means aguments");
  OVS_ASSERT(scales.size() == m.channels(),
             "image channel does not match scales aguments");
  OVS_ASSERT(CV_MAT_CN(type) == 1,
             "normalization for each channel require type of a single channel");
  switch (m.channels()) {
  case 1:
    return standard_score_normalize(m, type, means[0], scales[0]);
  case 2:
    return standard_score_normalize(m, type, {means[0], means[1]},
                                    {scales[0], scales[1]});
  case 3:
    return standard_score_normalize(m, type, {means[0], means[1], means[2]},
                                    {scales[0], scales[1], scales[2]});
  case 4:
    return standard_score_normalize(
        m, type, {means[0], means[1], means[2], means[4]},
        {scales[0], scales[1], scales[2], scales[4]});
  default:
    throw "support only 1 ot 4 channel image";
  }
}

inline cv::Mat standard_score_normalize(cv::Mat &m, int type, cv::Scalar means,
                                        cv::Scalar scales) {
  // convertTo formula is  Y = α*X + β
  // standard score is     Y = (X - μ)/σ
  // CV formula is         Y = (x/σ) + (-μ/σ)
  // ***************************************************
  std::vector<cv::Mat> by_channels(m.channels());
  cv::split(m, by_channels);
  for (int i = 0; i < by_channels.size(); i++) {
    m.convertTo(by_channels[i], type, 1 / scales[i], -means[i] / scales[i]);
  }
  cv::merge(by_channels, m);
  return m;
}

} // namespace pipeline::image::op
