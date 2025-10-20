#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <openvino/core/model.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <openvino/runtime/tensor.hpp>
#include <optional>
#include <vector>

#include "common/ov_types.hpp"
#include "common/ovs_assert.hpp"

namespace common = comfaix::common;

namespace segmentation::image::deeplearning::vit {

// @brief a process processing step before pass image to the model
// @detail the image should have a final data layout that require by the
// model. i.e, the result cv::Mat data is already in layout form of NCHW.
using PreProcessing = std::function<cv::Mat(const cv::Mat &src)>;

// @brief class for handling Vision Transformer model
class ViT {
public:
  ViT(std::string model_path, std::string device, std::string cache_dir,
      std::optional<PreProcessing> preprocessing = std::nullopt);

  // @brief create embedding data
  ov::Tensor embed(const cv::Mat &src);

private:
  const std::optional<PreProcessing> preprocessing;
  std::shared_ptr<ov::Model> model;
  ov::CompiledModel compiled_model;
  ov::InferRequest infer_request;
};

} // namespace segmentation::image::deeplearning::vit

// *****************************************
// *************** Dino V2 *****************
// *****************************************

namespace segmentation::image::deeplearning::vit::dino::v2 {

struct ResizeSize {
  int shortest_edge;
};

struct ImageProcessorConfig {
  cv::Size crop_size;
  bool do_center_crop;
  bool do_convert_rgb;
  bool do_normalize;
  bool do_rescale;
  bool do_resize;
  cv::Scalar image_mean;
  std::string image_processor_type;
  cv::Scalar image_std;
  int resample;
  double rescale_factor;
  ResizeSize size;
};

// @brief read the JSON of dino version 2 configuration
ImageProcessorConfig read_config(const std::string &path);

// @brief pre-processing handler for dino v2
PreProcessing create_preprocessing(const ImageProcessorConfig &config);

// @brief return a copy of global token
template <typename T> std::vector<T> global_token(ov::Tensor &tensor) {
  auto type = tensor.get_element_type();
  std::vector<T> results;
  common::dispatch_by_type(type, [&results, &tensor]<typename A>() {
    auto ss = std::is_same_v<A, T>;
    OVS_ASSERT(ss, "type is not the same");
    auto shape = tensor.get_shape();
    if (shape.size() == 3) {
      // first token data
      A *data = tensor.data<A>();
      results.reserve(shape[2]);
      std::copy(data, data + shape[2], std::back_inserter(results));
    }
  });
  return results;
}

} // namespace segmentation::image::deeplearning::vit::dino::v2
