#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <openvino/runtime/tensor.hpp>
#include <vector>

#include "common/segmentation.hpp"
#include "ocr/paddle/config.hpp"
#include "pipeline/batch.hpp"

namespace common = comfaix::common;

namespace ocr::paddle {

// @brief a record data use to store the state of batch processing.
struct Records {
  cv::Size size; // maximum size of image to current batch
  ov::Tensor tensor;
  // @brief reset all data to empty or zero
  void reset(cv::Size in = {}) {
    size.width = in.width;
    size.height = in.height;
  }
};

// @brief a detection class use to recognize text on an image.
class RecognizePreProcessing
    : public pipeline::IProcessing<Records, common::ImageSegmentionResult,
                                   ov::Tensor> {
public:
  // @brief create RecognizePreProcessing
  RecognizePreProcessing(const RecognizeConfig &config);

  // support move constructor
  RecognizePreProcessing(RecognizePreProcessing &&other) noexcept;

  // support move assigment
  RecognizePreProcessing &operator=(RecognizePreProcessing &&other) noexcept {
    return *this;
  }

  // @brief initialize process processing step. This function will be call when
  // a new batch started.
  Records batch_initialize(
      int start, int count,
      const std::vector<common::ImageSegmentionResult> &segments) override;

  // @brief handling each processing item
  common::ImageSegmentionResult
  batch_processing(int index, const cv::Mat &origin, Records &records,
                   common::ImageSegmentionResult &segment) override;

  // @brief call after all items has been processed.
  ov::Tensor batch_post_processing(
      Records &records, int start, int count,
      const std::vector<common::ImageSegmentionResult> &segments) override;

private:
  Records records;
  const RecognizeConfig config;
};

} // namespace ocr::paddle
