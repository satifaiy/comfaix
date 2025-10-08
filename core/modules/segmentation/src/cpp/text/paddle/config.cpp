#include <opencv2/core/types.hpp>
#include <openvino/core/layout.hpp>
#include <openvino/runtime/core.hpp>
#include <regex>
#include <simdjson.h>

#include "segmentation/text/paddle/config.hpp"

using namespace simdjson;

// @brief support for reading configuration of model version 5
namespace segmentation::text::paddle {

inline namespace v5 {

// @breif read the detection configuration from the model
DetectConfig read_detect_config(std::string path, std::string device) {
  DetectConfig config;
  config.model_path = std::filesystem::path(path).append("inference.xml");
  if (!std::filesystem::exists(config.model_path)) {
    config.model_path =
        config.model_path.parent_path().append("inference.onnx");
  }
  config.device = device;

  std::filesystem::path config_path(path);
  config_path.append("config.json");

  ondemand::parser parser;
  padded_string json = padded_string::load(config_path.generic_string());
  ondemand::document jconfig = parser.iterate(json);

  auto thresh = jconfig["PostProcess"];
  config.db_config.thresh = thresh["thresh"];
  config.db_config.box_thresh = thresh["box_thresh"];
  config.db_config.unclip_ratio = thresh["unclip_ratio"];
  config.db_config.max_candidates = thresh["max_candidates"].get<int>();

  config.block_size = cv::Size(32, 32);
  ondemand::array arr = jconfig["PreProcess"]["transform_ops"];
  for (ondemand::value ele : arr) {
    ondemand::object opobj = ele.get_object();
    auto result = opobj.find_field("NormalizeImage");
    if (result.error() == SUCCESS) {
      auto normalize = *result;
      auto means = normalize["mean"].get_array();
      int index = 0;
      for (auto mean : means) {
        config.means[index++] = mean.get<double>();
      }
      auto scales = normalize["std"].get_array();
      index = 0;
      for (auto scale : scales) {
        config.scales[index++] = scale.get<double>();
      }
      std::regex pattern("1*/255*");
      auto min_max = normalize["scale"];
      config.need_min_max_normalize =
          std::regex_match(std::string(min_max), pattern);
      auto order = std::string(normalize["order"]);
      std::transform(order.begin(), order.end(), order.begin(),
                     [](unsigned char c) { return ::toupper(c); });
      config.layout = ov::Layout(order);
      continue;
    }

    result = opobj.find_field("DetResizeForTest");
    if (result.error() == SUCCESS) {
      int max_value = ele["DetResizeForTest"]["resize_long"];
      config.max_width_height = cv::Size(max_value, max_value);
    }
  }
  return config;
}

} // namespace v5

} // namespace segmentation::text::paddle
