#include <filesystem>
#include <opencv2/core/types.hpp>
#include <openvino/core/layout.hpp>
#include <openvino/core/partial_shape.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>

namespace ocr::paddle {

// @brief a configuration to use for detecting text region using paddle detect
// models.
struct RecognizeConfig {
  std::filesystem::path model_path;
  cv::Scalar means;
  cv::Scalar scales;
  ov::Layout layout;
  ov::element::Type type;
  ov::PartialShape partial_shape;
  cv::Size max_width_height;
  int batch_number;
  std::vector<std::string> recognized_characters;
  std::string device;
};

// @brief support for reading configuration of model version 5
inline namespace v5 {
// @breif read the detection configuration from the model
RecognizeConfig read_recognize_config(std::string path, std::string device);
} // namespace v5

} // namespace ocr::paddle
