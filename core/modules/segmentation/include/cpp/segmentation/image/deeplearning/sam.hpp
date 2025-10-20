#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/core/descriptor/tensor.hpp>
#include <openvino/core/model.hpp>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/core/type/element_type_traits.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <openvino/runtime/tensor.hpp>
#include <optional>
#include <utility>
#include <vector>

#include "common/ov_types.hpp"
#include "common/segmentation.hpp"
#include "segmentation/image/classic/query.hpp"

namespace common = comfaix::common;

namespace segmentation::image::deeplearning::sam {

// @brief a result of segment that include polygon and a proper segment
using Result = comfaix::common::Result<cv::Mat, std::vector<cv::Point>>;

// @brief a result of segment that include polygon and a mask of object
// @detail the segment in the result is a reference with a bounding box to
// original image thus the ResultMask is should only be use where original
// image is stil valid. If you need a clone result then use Result instead.
using ResultMask = comfaix::common::Result<cv::Mat, std::vector<cv::Point>>;

// @brief a point for segment selection
struct Point {
  float x, y;
};

// @brief a box for segment selection
struct Box {
  float x1, y1;
  float x2, y2;
};

// @brief name and type of the tensor
struct TypeName {
  std::string name;
  ov::element::Type type;
  int cv_depth; // a cv coorrespond to ov type
};

// trai chekc for type name
template <typename T>
concept IsTypeName = std::is_same_v<T, TypeName>;

// @brief tensor input name
struct TensorName {
  TypeName batched_images;
  TypeName batched_coords;
  TypeName batched_labels;
  TypeName embedded_images;      // use by decoder only
  TypeName original_images_size; // use by decoder only
  TypeName output_masks;
  TypeName predictions;
};

// constant value for SAM model
const float LABEL_POINT = 1;
const float LABEL_BACKGROUND = 0;
const float LABEL_BOX_TOP_LEFT = 2;
const float LABEL_BOX_BOTTOM_RIGHT = 3;

// @brief create point and label tensor from points
template <typename P, typename L>
void build_point_dispatched(ov::Tensor &tpoints, ov::Tensor &tlabel,
                            const std::vector<Point> &points) {
  tlabel.set_shape(ov::Shape({1, 1, points.size()}));
  tpoints.set_shape(ov::Shape({1, 1, points.size(), 2}));
  auto ldata = tlabel.data<L>();
  auto pdata = tpoints.data<P>();
  for (const auto p : points) {
    *ldata = L(LABEL_POINT);
    ldata++;
    *pdata = P(p.x);
    *(++pdata) = P(p.y);
    pdata++;
  }
}

inline void build_point_tensor(ov::Tensor &tpoints, ov::Tensor &tlabel,
                               const std::vector<Point> &points) {
  common::dispatch_by_type(tlabel.get_element_type(), [&tpoints, &tlabel,
                                                       points]<typename L>() {
    common::dispatch_by_type(
        tpoints.get_element_type(), [&tpoints, &tlabel, points]<typename P>() {
          build_point_dispatched<P, L>(tpoints, tlabel, points);
        });
  });
}

inline void build_anything_points_tensor(ov::Tensor &tpoints,
                                         ov::Tensor &tlabel, int width,
                                         int height, const cv::Size &grid) {
  common::dispatch_by_type(
      tlabel.get_element_type(),
      [&tpoints, &tlabel, width, height, grid]<typename L>() {
        common::dispatch_by_type(
            tpoints.get_element_type(),
            [&tpoints, &tlabel, width, height, grid]<typename P>() {
              int count = grid.width * grid.height;
              tlabel.set_shape(ov::Shape({1, 1, size_t(count)}));
              tpoints.set_shape(ov::Shape({1, 1, size_t(count), 2}));
              auto ldata = tlabel.data<L>();
              auto pdata = tpoints.data<P>();
              for (int x = 0; x < grid.width; x++) {
                for (int y = 0; y < grid.height; y++) {
                  *ldata = L(LABEL_POINT);
                  ldata++;
                  *pdata = P(0.5 + (float(x) / grid.width) * width);
                  *(++pdata) = P(0.5 + (float(y) / grid.height) * height);
                  pdata++;
                }
              }
            });
      });
}

// @brief create point and label tensor from points
template <typename P, typename L>
static void build_box_dispatched(ov::Tensor &tboxes, ov::Tensor &tlabel,
                                 const std::vector<Box> &boxes) {
  tlabel.set_shape(ov::Shape({1, 1, boxes.size() * 2}));
  tboxes.set_shape(ov::Shape({1, 1, boxes.size() * 2, 2}));
  auto ldata = tlabel.data<L>();
  auto bdata = tboxes.data<P>();
  for (const auto b : boxes) {
    *ldata = L(LABEL_BOX_TOP_LEFT);
    *(++ldata) = L(LABEL_BOX_BOTTOM_RIGHT);
    ldata++;
    bdata[0] = std::min(P(b.x1), P(b.x2));
    bdata[2] = std::max(P(b.x1), P(b.x2));
    bdata[1] = std::min(P(b.y1), P(b.y2));
    bdata[3] = std::max(P(b.y1), P(b.y2));
    bdata += 4;
  }
}

inline void build_box_tensor(ov::Tensor &tboxes, ov::Tensor &tlabel,
                             const std::vector<Box> &boxes) {
  common::dispatch_by_type(
      tlabel.get_element_type(), [&tboxes, &tlabel, boxes]<typename L>() {
        common::dispatch_by_type(
            tboxes.get_element_type(), [&tboxes, &tlabel, boxes]<typename P>() {
              build_box_dispatched<P, L>(tboxes, tlabel, boxes);
            });
      });
}

// @brief inference class use for reusable mechanism in SAM model family.
class Inference {
public:
  // @brief Inference constructure
  Inference(const cv::Mat &src, const std::optional<ov::Tensor> &embedd,
            const TensorName &input_names, ov::InferRequest infer_request);

  // @brief segmenting src
  // @detail if points and boxes is not porvided then segment will try segment
  // everything
  std::vector<Result> segment(const std::vector<Point> points = {},
                              const std::vector<Box> boxes = {},
                              const cv::Size &grid_anything = {32, 32}) {
    return segmentation<Result>(points, boxes, grid_anything, false);
  }

  // @brief segmenting src
  // @detail if points and boxes is not porvided then segment will try segment
  // everything
  std::vector<ResultMask> segment_mask(const std::vector<Point> points = {},
                                       const std::vector<Box> boxes = {},
                                       const cv::Size &grid_anything = {32,
                                                                        32}) {
    return segmentation<ResultMask>(points, boxes, grid_anything, true);
  }

private:
  const cv::Mat src;
  const std::optional<ov::Tensor> embed;
  const TensorName &names;
  ov::InferRequest infer_request;

  // callback when segment found
  using callback = std::function<void(
      const cv::Mat &, const std::vector<cv::Point> &, float score)>;

  // @brief segmentation using SAM family models
  template <typename R>
  std::vector<R> segmentation(const std::vector<Point> points = {},
                              const std::vector<Box> boxes = {},
                              const cv::Size &grid_anything = {0, 0},
                              const bool mask = true) {
    std::vector<R> results;
    callback cb = [&results, mask](const cv::Mat &src,
                                   const std::vector<cv::Point> &points,
                                   float score) {
      if (mask) {
        cv::Rect rect = cv::boundingRect(points);
        results.emplace_back(ResultMask{
            src(rect),
            points,
            score,
        });
      } else {
        cv::Mat cropped;
        classic::crop(src, cropped, points);
        results.emplace_back(Result{
            cropped,
            points,
            score,
        });
      }
    };
    segmentation(cb, points, boxes, grid_anything);
    return results;
  }

  // @brief segmenting the image src
  void segmentation(callback found, const std::vector<Point> points = {},
                    const std::vector<Box> boxes = {},
                    const cv::Size &grid_anything = {0, 0});
};

// @brief SAM a class provide handling for SAM family model
class SAM {
public:
  // @brief create SAM like class for segmentation
  // @param names three inputs tensor name in order. First is image input,
  // second is coordinate input and third is labels indcate whether coordinate
  // is a box or point.
  SAM(std::string model_path, std::string device,
      std::filesystem::path cache_dir = "",
      const TensorName &names = {{"batched_images", ov::element::f32},
                                 {"batched_point_coords", ov::element::f32},
                                 {"batched_point_labels", ov::element::f32},
                                 {"image_embeddings", ov::element::f32},
                                 {"orig_im_size", ov::element::i64},
                                 {"output_masks", ov::element::u8},
                                 {"iou_predictions", ov::element::f32}});

  // @brief create SAM like class for segmentation
  // @detail useful for frequently segmenting from same image multiple time as
  // the encode happen only one time resulting in a very quick subsequence
  // segmentation.
  // @param names three inputs tensor name in order. First is image input,
  // second is coordinate input and third is labels indcate whether coordinate
  // is a box or point.
  SAM(std::string model_encoder, std::string model_decoder, std::string device,
      std::filesystem::path cache_dir = "",
      const TensorName &names = {{"batched_images", ov::element::f32},
                                 {"batched_point_coords", ov::element::f32},
                                 {"batched_point_labels", ov::element::f32},
                                 {"image_embeddings", ov::element::f32},
                                 {"orig_im_size", ov::element::i64},
                                 {"output_masks", ov::element::u8},
                                 {"iou_predictions", ov::element::f32}});

  // @brief create source inference from source image.
  const Inference create_infer(const cv::Mat &src);

private:
  TensorName names;
  std::shared_ptr<ov::Model> model;
  ov::CompiledModel compiled_model;
  std::optional<std::shared_ptr<ov::Model>> decode_model;
  std::optional<ov::CompiledModel> decode_compiled_model;

  // @brief read type from tensor input or output and assign correspond type to
  // the arguments TypeName.
  template <IsTypeName... T>
  static void read_types(const ov::CompiledModel single, bool is_input,
                         T &...tname) {
    auto func = [single, is_input](TypeName &&x) {
      ov::Output<const ov::Node> iop;
      if (is_input)
        iop = single.input(x.name);
      else
        iop = single.output(x.name);

      x.type = iop.get_element_type();
      switch (x.type) {
      case ov::element::f16:
        x.cv_depth = CV_16F;
        break;
      case ov::element::f32:
        x.cv_depth = CV_32F;
        break;
      case ov::element::f64:
        x.cv_depth = CV_64F;
        break;
      case ov::element::u8:
        x.cv_depth = CV_8U;
        break;
      case ov::element::u16:
        x.cv_depth = CV_16U;
        break;
      case ov::element::i8:
        x.cv_depth = CV_8S;
        break;
      case ov::element::i16:
        x.cv_depth = CV_16S;
        break;
      case ov::element::i32:
        x.cv_depth = CV_32S;
        break;
      default:
        // fallback to raw byte data. it's not correct for example u64 or i64
        // throwing instead??
        x.cv_depth = CV_8U;
      }
    };
    (func(std::forward<T>(tname)), ...);
  }
};

} // namespace segmentation::image::deeplearning::sam
