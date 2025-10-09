#include <vector>

#include "common/segmentation.hpp"

namespace common = comfaix::common;

namespace ocr::paddle {

// @brief abstraction class that provide method to recognize text on the image.
template <typename S, typename R> class IRecognize {
public:
  virtual ~IRecognize() = default;

  // @brief recognize the text on the given image matrix m.
  virtual std::vector<common::Result<S, R>> recognize(const cv::Mat &m) = 0;
};

} // namespace ocr::paddle
