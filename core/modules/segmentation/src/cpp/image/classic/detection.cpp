#include <functional>
#include <oneapi/tbb.h>
#include <oneapi/tbb/null_mutex.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_for_each.h>
#include <oneapi/tbb/queuing_mutex.h>
#include <oneapi/tbb/spin_mutex.h>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_group.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <vector>

#include "segmentation/image/classic/detection.hpp"
#include "segmentation/image/classic/edge.hpp"
#include "segmentation/image/classic/query.hpp"

namespace segmentation::image::classic {

// @brief implement detection constructor
Detection::Detection(const EdgeDetection edge_detection,
                     const Similarity similarity)
    : edge_detection(std::move(edge_detection)),
      similarity(std::move(similarity)) {}

// @brief find and extract the image that is similar to the input queries.
std::vector<Result> Detection::detect(const cv::Mat &source,
                                      const Queries &queries) {
  auto edge = this->edge_detection(source);
  if (edge.type() != CV_8UC1)
    cv::cvtColor(edge, edge, cv::COLOR_BGR2GRAY);
  cv::threshold(edge, edge, 128, 255, cv::THRESH_BINARY);

  std::vector<std::vector<cv::Point>> search_contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(edge, search_contours, hierarchy, cv::RETR_TREE,
                   cv::CHAIN_APPROX_SIMPLE);

  tbb::spin_mutex mutex;
  std::vector<Result> results;
  results.reserve(search_contours.size());

  std::optional<std::reference_wrapper<Queries>> noquery = std::nullopt;
  std::function<void(int)> dispatch;
  dispatch = [source, search_contours, queries, hierarchy, &noquery, &dispatch,
              &results, &mutex, this](int sidx) {
    // start from root
    auto sc = search_contours[sidx];
    auto area = cv::contourArea(sc);

    if (queries.min_area <= area && area <= queries.max_area) {
      ClassicAttribute target_attr = {};
      std::optional<std::reference_wrapper<ClassicAttribute>> opt_attr =
          std::ref(target_attr);
      opt_attr->get().geometry.area = area;
      Detection::shape_attr(sc, 0, opt_attr, noquery);

      cv::Mat cropped;

      // check shape
      for (int qidx = 0; qidx < queries.data.size(); qidx++) {
        // area is less than or greater than acceptable range
        if (opt_attr->get().geometry.area <
                queries.data[qidx].attribute.geometry.min_area ||
            queries.data[qidx].attribute.geometry.max_area <
                opt_attr->get().geometry.area) {
          continue;
        }

        // check contour shape similarity
        const auto attr = queries.data[qidx].attribute;
        if (!attr.similar(opt_attr->get())) {
          continue;
        }

        if (cropped.rows == 0 && cropped.cols == 0) {
          if (!crop(source, cropped, sc)) {
            // crop is viable
            break;
          }
        }

        // compare atual segment
        double score = this->similarity(queries.data[qidx].data, cropped);
        if (score > queries.data[qidx].min_score) {
          // append result
          mutex.lock();
          results.push_back({
              {cropped, qidx},
              sc,
              float(score),
          });
          mutex.unlock();
          return;
        }
      }
    }

    // check child
    int child_index = hierarchy[sidx][2];
    if (child_index >= 0) {
      tbb::task_group child;
      while (child_index >= 0) {
        int index = child_index;
        child.run([index, dispatch] { dispatch(index); });
        child_index = hierarchy[child_index][0];
      }
      child.wait();
    }
  };

  tbb::task_group parent;
  for (int i = 0; i < search_contours.size(); i++) {
    if (hierarchy[i][3] == -1) {
      int index = i;
      parent.run([index, dispatch] { dispatch(index); });
    }
  }
  parent.wait();
  results.shrink_to_fit();
  return results;
}

// @brief find and extract the image that is similar to the input queries.
std::vector<Result> Detection::detect(const cv::Mat &source,
                                      const std::vector<cv::Mat> &queries,
                                      const std::vector<double> &min_scores,
                                      const uint32_t tolerance) {
  return detect(source, build_queries(queries, min_scores, tolerance));
}

} // namespace segmentation::image::classic
