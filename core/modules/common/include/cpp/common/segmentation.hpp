#include <vector>

// @brief a common data type
namespace comfaix::common {

// @brief result of segmentations
template <typename S, typename R> struct Result {
  // @brief a segment data. It be a copy/clone or a reference to original data.
  S segment;
  // @brief a region of interest. It's a geometry representation of a segment.
  // i.e polygon.
  R roi;
  // @brief a matching score of the segment.
  float score;
};

// @brief abstraction class provide common method to segment an input.
template <typename S, typename R> class Segmentation {
public:
  // @brief default destructor
  virtual ~Segmentation() = default;

  // @brief segment the input source and produce result into results.
  virtual void segment(const S &source, std::vector<Result<S, R>> &results) = 0;
};

} // namespace comfaix::common
