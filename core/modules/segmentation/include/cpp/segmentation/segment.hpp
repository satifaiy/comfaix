#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

#include "common/segmentation.hpp"

namespace common = comfaix::common;

namespace segmentation {

// @brief abstraction class that provide method to segment the image.
template <typename S, typename R> class ISegmentation {
public:
  virtual ~ISegmentation() = default;

  // @brief segmenting the image and set the result in result argument.
  virtual std::vector<common::Result<S, R>> segment(const cv::Mat &m) = 0;
};

// @brief a grid that represent column and row base on image source
class GridVisualization {
public:

private:
    const int rows;
    const int columns;
    const std::unique_ptr<int[]> grid_data;

    // @brief constructor to create GridVisualization
    GridVisualization(int rows, int columns, std::unique_ptr<int[]> &grid_data);
};

} // namespace segmentation
