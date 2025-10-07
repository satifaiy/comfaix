#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <type_traits>

#include "common/ovs_assert.hpp"

namespace pipeline::image::op {

// @brief short alias for checking number type
template <typename T>
using NUMBER = typename std::enable_if<
    std::is_integral<T>::value || std::is_floating_point<T>::value, T>::type;

// @brief rearrange a 2 channel data into planar base
template <typename T> inline NUMBER<T> *permute_c2(cv::Mat &m, T *data) {
  int each = m.cols * m.rows;
  T *ch1 = (T *)data;
  T *ch2 = (T *)data + each;
  T *raw = (T *)m.data;
  for (int i = 0; i < each; i++) {
    *ch1++ = *raw++;
    *ch2++ = *raw++;
  }
  return data;
}

// @brief rearrange a 3 channel data into planar base
template <typename T> inline NUMBER<T> *permute_c3(cv::Mat &m, T *data) {
  int each = m.cols * m.rows;
  T *ch1 = (T *)data;
  T *ch2 = ch1 + each;
  T *ch3 = ch2 + each;
  T *raw = (T *)m.data;
  for (int i = 0; i < each; i++) {
    *ch1++ = *raw++;
    *ch2++ = *raw++;
    *ch3++ = *raw++;
  }
  return data;
}

// @brief rearrange a 4 channel data into planar base
template <typename T> inline NUMBER<T> *permute_c4(cv::Mat &m, T *data) {
  int each = m.cols * m.rows;
  T *ch1 = (T *)data;
  T *ch2 = ch1 + each;
  T *ch3 = ch2 + each;
  T *ch4 = ch3 + each;
  T *raw = (T *)m.data;
  for (int i = 0; i < each; i++) {
    *ch1++ = *raw++;
    *ch2++ = *raw++;
    *ch3++ = *raw++;
    *ch4++ = *raw++;
  }
  return data;
}

// @brief flatten the channel
template <typename T> inline NUMBER<T> *permute(cv::Mat &m, T *data) {
  OVS_ASSERT(0 < m.channels() && m.channels() < 5,
             "permute support image with 1 to 4 channels only, got ", m.channels());
  switch (m.channels()) {
  case 1:
    memcpy(data, (T *)m.data, m.cols * m.rows);
    return data;
  case 2:
    return permute_c2(m, data);
  case 3:
    return permute_c3(m, data);
  case 4:
    return permute_c4(m, data);
  default:
    throw "unsupported";
  }
}

} // namespace pipeline::image::op
