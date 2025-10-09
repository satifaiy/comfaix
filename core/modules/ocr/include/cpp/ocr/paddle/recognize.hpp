#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <openvino/runtime/tensor.hpp>
#include <string>
#include <vector>

#include "common/segmentation.hpp"
#include "ocr/paddle/pre_processing.hpp"
#include "ocr/recognize.hpp"
#include "segmentation/segment.hpp"

namespace ocr::paddle {

using Batch = pipeline::BatchProcessor<Records, common::ImageSegmentionResult,
                                       ov::Tensor>;

using Segmentation =
    segmentation::ISegmentation<cv::Mat, std::vector<cv::Point2f>>;

// @brief a detection class use to recognize text on an image.
class Recognize : public IRecognize<std::string, std::vector<cv::Point2f>> {
public:
  // @brief create Recognize with the given model path and device.
  // @param path is the path to the text detection model.
  // @param device is the openvino backend
  Recognize(std::string path, std::unique_ptr<Segmentation> segmentation,
            std::string device);

  // @brief create Recognize with the given model config and device.
  // @param detect model configuration
  // @param device is the openvino backend
  Recognize(const RecognizeConfig &config,
            std::unique_ptr<Segmentation> segmentation);

  // @brief implement ISegementation.
  std::vector<common::TextRecognitionResult>
  recognize(const cv::Mat &m) override;

private:
  template <class ForwardIterator>
  inline static size_t max_score_index(ForwardIterator first,
                                       ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
  }

  const RecognizeConfig config;
  std::shared_ptr<ov::Model> model;
  ov::CompiledModel compiled_model;
  ov::InferRequest infer_request;
  Batch batch;
  std::unique_ptr<Segmentation> segmentation;
};

} // namespace ocr::paddle
