#include <opencv2/core/mat.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/tensor.hpp>

#include "segmentation/image/deeplearning/vit.hpp"

namespace segmentation::image::deeplearning::vit {

ViT::ViT(std::string model_path, std::string device, std::string cache_dir,
         std::optional<PreProcessing> preprocessing)
    : preprocessing(preprocessing) {
  ov::Core core;
  if (!cache_dir.empty())
    core.set_property(ov::cache_dir(cache_dir));
  this->model = core.read_model(model_path);
  this->compiled_model = core.compile_model(this->model);
  this->infer_request = this->compiled_model.create_infer_request();
}

// @brief create embedding data
ov::Tensor ViT::embed(const cv::Mat &src) {
  cv::Mat input = src;
  ov::Shape shape = {1, size_t(src.channels()), size_t(src.rows),
                     size_t(src.cols)};
  if (this->preprocessing.has_value()) {
    input = this->preprocessing.value()(src);
    auto size = input.size;
    shape[1] = size[1];
    shape[2] = size[2];
    shape[3] = size[3];
  }
  ov::Tensor input_tensor(ov::element::f32, shape, input.data);
  this->infer_request.set_input_tensor(input_tensor);
  this->infer_request.infer();
  return this->infer_request.get_output_tensor();
}

} // namespace segmentation::image::deeplearning::vit
