#pragma once

#include <cstdint>
#include <functional>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <optional>
#include <vector>

#include "common/segmentation.hpp"
#include "segmentation/image/classic/edge.hpp"
#include "segmentation/image/classic/embed.hpp"
#include "segmentation/image/classic/query.hpp"

namespace common = comfaix::common;

namespace segmentation::image::classic {

// @brief result of founded segmentation
struct ResultSegment {
  // segmentation of image
  cv::Mat segment;

  // index of query
  int index;
};

// @brief result of detected query
using Result = common::Result<ResultSegment, std::vector<cv::Point>>;

// @brief classic detect using simple basic recognizable shapes
class Detection {
public:
  Detection(const EdgeDetection edge_detection, const Similarity similarity);

  // @brief find and extract the image that is similar to the input queries.
  std::vector<Result> detect(const cv::Mat &source, const Queries &queries);

  // @brief find and extract the image that is similar to the input queries.
  std::vector<Result> detect(const cv::Mat &source,
                             const std::vector<cv::Mat> &queries,
                             const std::vector<double> &min_scores = {0.90},
                             uint32_t tolerance = 10);

  // @brief query with default option
  Queries build_queries(const std::vector<cv::Mat> &src,
                        const std::vector<double> &min_scores = {0.90},
                        const uint32_t tolerance = 10);

  // @brief shape of the points with need extra compute data
  static ShapeAttr shape_attr(const std::vector<cv::Point> &points,
                              const uint32_t &tolerance = 0);

  // @brief shape of the points
  static ShapeAttr
  shape_attr(const std::vector<cv::Point> &points, const uint32_t &tolerance,
             std::optional<std::reference_wrapper<ClassicAttribute>> &attr,
             std::optional<std::reference_wrapper<Queries>> &queries);

private:
  const EdgeDetection edge_detection;
  const Similarity similarity;
};

} // namespace segmentation::image::classic
