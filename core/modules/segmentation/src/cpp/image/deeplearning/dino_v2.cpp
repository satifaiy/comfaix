#include <cstdlib>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <simdjson.h>

#include "pipeline/image/normalize_op.hpp"
#include "pipeline/image/permute_op.hpp"
#include "segmentation/image/deeplearning/vit.hpp"

using namespace simdjson;

namespace op = pipeline::image::op;

namespace segmentation::image::deeplearning::vit::dino::v2 {

// @brief read the JSON of dino version 2 configuration
ImageProcessorConfig read_config(const std::string &path) {
  ImageProcessorConfig config;
  ondemand::parser parser;
  padded_string json = padded_string::load(path);
  ondemand::document doc = parser.iterate(json);

  config.do_center_crop = doc["do_center_crop"].get_bool();
  config.do_convert_rgb = doc["do_convert_rgb"].get_bool();
  config.do_normalize = doc["do_normalize"].get_bool();
  config.do_rescale = doc["do_rescale"].get_bool();
  config.do_resize = doc["do_resize"].get_bool();
  config.resample = doc["resample"].get_int64();
  switch (config.resample) {
  case 0:
    config.resample = cv::InterpolationFlags::INTER_NEAREST;
    break;
  case 1:
    config.resample = cv::InterpolationFlags::INTER_LANCZOS4;
    break;
  case 2:
    config.resample = cv::InterpolationFlags::INTER_LINEAR;
    break;
  case 3:
    config.resample = cv::InterpolationFlags::INTER_CUBIC;
    break;
  }

  config.rescale_factor = doc["rescale_factor"].get_double();

  ondemand::object crop_size_obj = doc["crop_size"].get_object();
  config.crop_size.height = crop_size_obj["height"].get_int64();
  config.crop_size.width = crop_size_obj["width"].get_int64();

  ondemand::object size_obj = doc["size"].get_object();
  config.size.shortest_edge = size_obj["shortest_edge"].get_int64();

  ondemand::array mean_arr = doc["image_mean"].get_array();
  int index = 0;
  for (auto element : mean_arr) {
    config.image_mean[index++] = element.get_double();
  }

  ondemand::array std_arr = doc["image_std"].get_array();
  index = 0;
  for (auto element : std_arr) {
    config.image_std[index++] = element.get_double();
  }
  return config;
}

// @brief pre-processing handler for dino v2
PreProcessing create_preprocessing(const ImageProcessorConfig &config) {
  return [config](const cv::Mat &src) -> cv::Mat {
    cv::Mat target = src;
    if (config.do_resize && (src.cols != config.crop_size.width ||
                             src.rows != config.crop_size.height)) {
      // resize keep aspect ratio
      cv::Size resize;
      if (src.cols < src.rows) {
        // ratio scale by height
        resize.width = config.size.shortest_edge;
        resize.height = src.rows * config.size.shortest_edge / src.cols;
      } else {
        // ratio scale by height
        resize.height = config.size.shortest_edge;
        resize.width = src.cols * config.size.shortest_edge / src.rows;
      }
      cv::resize(src, target, resize, 0, 0, config.resample);
    }

    if (config.do_center_crop) {
      cv::Rect crop((target.cols - config.crop_size.width) / 2,
                    (target.rows - config.crop_size.height) / 2,
                    config.crop_size.width, config.crop_size.height);
      target = target(crop).clone();
    }

    if (config.do_convert_rgb)
      cv::cvtColor(target, target, cv::COLOR_BGR2RGB);

    int depth = target.depth();
    if (config.do_rescale) {
      depth = CV_32F;
      target = op::min_max_normalize(target, CV_32F);
    }

    if (config.do_normalize) {
      depth = CV_32F;
      target = op::standard_score_normalize(target, CV_32F, config.image_mean,
                                            config.image_std);
    }

    int dims[] = {1, target.channels(), target.rows, target.cols};
    cv::Mat permute(4, dims, depth);
    if (depth == CV_32F) {
      op::permute(target, reinterpret_cast<float *>(permute.data));
    } else {
      op::permute(target, permute.data);
    }
    return permute;
  };
}

} // namespace segmentation::image::deeplearning::vit::dino::v2
