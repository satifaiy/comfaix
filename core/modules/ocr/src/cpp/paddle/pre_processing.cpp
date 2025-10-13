#include <algorithm>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgcodecs.hpp>
#include <openvino/runtime/tensor.hpp>

#include "ocr/paddle/pre_processing.hpp"
#include "pipeline/image/normalize_op.hpp"
#include "pipeline/image/padding_op.hpp"
#include "pipeline/image/permute_op.hpp"
#include "pipeline/image/resize_op.hpp"

namespace pimg = pipeline::image::op;

namespace ocr::paddle {

// @brief create RecognizePreProcessing
RecognizePreProcessing::RecognizePreProcessing(const RecognizeConfig &config)
    : config(config) {}

// @brief initialize process processing step. This function will be call when
// a new batch started.
Records RecognizePreProcessing::batch_initialize(
    int start, int count,
    const std::vector<common::ImageSegmentionResult> &segments) {
  this->records.reset({0, int(this->config.partial_shape[2].get_length())});
  auto height = this->records.size.height;
  for (const auto seg : segments) {
    auto resized_width = (seg.segment.cols * height) / seg.segment.rows;
    this->records.size.width =
        std::max(resized_width, this->records.size.width);
  }
  ov::Shape shape = {size_t(count), 3, size_t(this->records.size.height),
                     size_t(this->records.size.width)};
  this->records.tensor = ov::Tensor(ov::element::f32, shape);
  return this->records;
};

// @brief handling each processing item
common::ImageSegmentionResult RecognizePreProcessing::batch_processing(
    int index, const cv::Mat &origin, Records &records,
    common::ImageSegmentionResult &segment) {
  cv::Mat seg = pimg::resize_fit_height(segment.segment, records.size.height,
                                        cv::InterpolationFlags::INTER_LINEAR);
  if (seg.cols < records.size.width) {
    seg =
        pimg::padding_right(seg, records.size.width - seg.cols,
                            cv::BorderTypes::BORDER_CONSTANT, {127, 127, 127});
  }
  seg = pimg::min_max_normalize(seg, CV_MAKETYPE(CV_32F, seg.channels()));
  seg = pimg::standard_score_normalize(seg, CV_32FC1, this->config.means,
                                       this->config.scales);
  auto data = this->records.tensor.data<float>();
  data += index * seg.channels() * records.size.area();
  pimg::permute(seg, data);
  return segment;
};

// @brief call after all items has been processed.
ov::Tensor RecognizePreProcessing::batch_post_processing(
    Records &records, int start, int count,
    const std::vector<common::ImageSegmentionResult> &segments) {
  return records.tensor;
};

} // namespace ocr::paddle
