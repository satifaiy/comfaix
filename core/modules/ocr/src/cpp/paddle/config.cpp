#include <filesystem>
#include <opencv2/core/types.hpp>
#include <openvino/core/dimension.hpp>
#include <openvino/core/layout.hpp>
#include <openvino/core/type/element_type.hpp>
#include <simdjson.h>
#include <string>

#include "ocr/paddle/config.hpp"

using namespace simdjson;

namespace ocr::paddle {

// @brief support for reading configuration of model version 5
inline namespace v5 {

// @breif read the detection configuration from the model
RecognizeConfig read_recognize_config(std::string path, std::string device) {
  RecognizeConfig config;
  config.model_path = std::filesystem::path(path).append("inference.xml");
  if (!std::filesystem::exists(config.model_path)) {
    config.model_path =
        config.model_path.parent_path().append("inference.onnx");
  }
  config.device = device;
  config.layout = ov::Layout("NCHW");
  config.means = {0.5f, 0.5f, 0.5f};
  config.scales = {0.5f, 0.5f, 0.5f};
  config.batch_number = 8;
  config.type = ov::element::f32;

  std::filesystem::path config_path(path);
  config_path.append("config.json");

  ondemand::parser parser;
  padded_string json = padded_string::load(config_path.generic_string());
  ondemand::document jconfig = parser.iterate(json);

  ondemand::array arr = jconfig["PreProcess"]["transform_ops"];
  for (ondemand::value ele : arr) {
    ondemand::object opobj = ele.get_object();
    auto result = opobj.find_field("RecResizeImg");
    if (result.error() == SUCCESS) {
      auto resize_img = *result;
      auto shape = resize_img["image_shape"].get_array();
      int index = 0;
      config.partial_shape.push_back(ov::Dimension::dynamic());
      for (auto val : shape) {
        if (index == 0 || index == 1) {
          index++;
          config.partial_shape.push_back(ov::Dimension(val.get<int>()));
        } else {
          break;
        }
      }
      config.max_width_height.height = config.partial_shape[2].get_length();
      config.max_width_height.width = 0;
      config.partial_shape.push_back(ov::Dimension::dynamic());
      break;
    }
  }

  // read eligible character as string
  jconfig = parser.iterate(json);
  arr = jconfig["PostProcess"]["character_dict"].get_array();
  for (ondemand::value ele : arr) {
    config.recognized_characters.push_back(ele.get<std::string>());
  }
  config.recognized_characters.insert(config.recognized_characters.begin(),
                                      "#");
  config.recognized_characters.push_back(" ");
  return config;
}

} // namespace v5

} // namespace ocr::paddle
