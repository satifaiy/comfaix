#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/core/layout.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/properties.hpp>
#include <openvino/runtime/tensor.hpp>
#include <vector>

#include "common/ovs_assert.hpp"
#include "segmentation/image/classic/edge.hpp"
#include "segmentation/image/deeplearning/sam.hpp"

namespace segmentation::image::deeplearning::sam {

// @brief create SAM like class for segmentation
// @param names three inputs tensor name in order. First is image input,
// second is coordinate input and third is labels indcate whether coordinate
// is a box or point.
SAM::SAM(std::string model_path, std::string device,
         std::filesystem::path cache_dir, const TensorName &names) {
  ov::Core core;
  if (!cache_dir.empty())
    core.set_property(ov::cache_dir(cache_dir));
  this->model = core.read_model(model_path);
  this->compiled_model = core.compile_model(this->model, device);
  this->names = names;

  read_types(this->compiled_model, true, this->names.batched_images,
             this->names.batched_coords, this->names.batched_labels);

  read_types(this->compiled_model, false, this->names.output_masks,
             this->names.predictions);
}

// @brief create SAM like class for segmentation
// @detail useful for frequently segmenting from same image multiple time as
// the encode happen only one time resulting in a very quick subsequence
// segmentation.
// @param names three inputs tensor name in order. First is image input,
// second is coordinate input and third is labels indcate whether coordinate
// is a box or point.
SAM::SAM(std::string model_encoder, std::string model_decoder,
         std::string device, std::filesystem::path cache_dir,
         const TensorName &names) {
  ov::Core core;
  if (!cache_dir.empty())
    core.set_property(ov::cache_dir(cache_dir));
  this->model = core.read_model(model_encoder);
  this->compiled_model = core.compile_model(this->model, device);

  this->decode_model = core.read_model(model_decoder);
  this->decode_compiled_model =
      core.compile_model(this->decode_model.value(), device);
  this->names = names;

  read_types(this->compiled_model, true, this->names.batched_images);

  auto decoder = this->decode_compiled_model.value();
  read_types(decoder, true, this->names.embedded_images,
             this->names.batched_coords, this->names.batched_labels,
             this->names.original_images_size);
  read_types(decoder, false, this->names.output_masks, this->names.predictions);
}

} // namespace segmentation::image::deeplearning::sam

// ****************************************************************
// ******************* Inference Implementation *******************
// ****************************************************************

namespace segmentation::image::deeplearning::sam {

// @brief create source inference from source image.
const Inference SAM::create_infer(const cv::Mat &src) {
  if (this->decode_compiled_model.has_value()) {
    // we're in encode/decode model
    auto infer_request = this->compiled_model.create_infer_request();
    cv::Mat blob = cv::dnn::blobFromImage(src, 1 / 255.0, cv::Size(),
                                          cv::Scalar(), true, false, CV_32F);
    auto data = (float *)blob.data;
    ov::Shape intput_shape = {1, 3, static_cast<size_t>(src.rows),
                              static_cast<size_t>(src.cols)};
    ov::Tensor input_tensor(this->names.batched_images.type, intput_shape,
                            data);
    infer_request.set_tensor(this->names.batched_images.name, input_tensor);
    infer_request.infer();
    return Inference(src, std::move(infer_request.get_output_tensor()),
                     this->names,
                     this->decode_compiled_model->create_infer_request());
  } else {
    return Inference(src, std::nullopt, this->names,
                     this->compiled_model.create_infer_request());
  }
}

// @brief Inference constructure
Inference::Inference(const cv::Mat &src, const std::optional<ov::Tensor> &embed,
                     const TensorName &names, ov::InferRequest infer_request)
    : src(src), embed(embed), names(names), infer_request(infer_request) {}

// @brief segmenting the image src
void Inference::segmentation(callback found, const std::vector<Point> points,
                             const std::vector<Box> boxes,
                             const cv::Size &grid_anything) {
  ov::Tensor tpb;    // point or box tensor
  ov::Tensor tlabel; // label tensor
  if (points.size() == 0 && boxes.size() == 0) {
    // create grid point for segment anything
    tlabel = ov::Tensor(this->names.batched_labels.type, ov::Shape());
    tpb = ov::Tensor(this->names.batched_coords.type, ov::Shape());
    build_anything_points_tensor(tpb, tlabel, this->src.cols, this->src.rows,
                                 grid_anything);
  } else {
    OVS_ASSERT((points.size() == 0) != (boxes.size() == 0),
               "points or boxes must be provided");
    if (points.size() > 0) {
      tlabel = ov::Tensor(this->names.batched_labels.type, ov::Shape());
      tpb = ov::Tensor(this->names.batched_coords.type, ov::Shape());
      build_point_tensor(tpb, tlabel, points);
    } else {
      tlabel = ov::Tensor(this->names.batched_labels.type, ov::Shape());
      tpb = ov::Tensor(this->names.batched_coords.type, ov::Shape());
      build_box_tensor(tpb, tlabel, boxes);
    }
  }

  if (!this->embed.has_value()) {
    cv::Mat blob = cv::dnn::blobFromImage(this->src, 1 / 255.0, cv::Size(),
                                          cv::Scalar(), true, false, CV_32F);
    auto data = (float *)blob.data;
    ov::Shape image_shape = {1, size_t(this->src.channels()),
                             static_cast<size_t>(src.rows),
                             static_cast<size_t>(src.cols)};

    ov::Tensor image_tensor(this->names.batched_images.type, image_shape, data);

    this->infer_request.set_tensor(this->names.batched_images.name,
                                   image_tensor);
    this->infer_request.set_tensor(this->names.batched_coords.name, tpb);
    this->infer_request.set_tensor(this->names.batched_labels.name, tlabel);
  } else {
    this->infer_request.set_tensor(this->names.embedded_images.name,
                                   this->embed.value());
    this->infer_request.set_tensor(this->names.batched_coords.name, tpb);
    this->infer_request.set_tensor(this->names.batched_labels.name, tlabel);

    ov::Shape shape = {2};
    ov::Tensor original_size(this->names.original_images_size.type, shape);
    auto data = original_size.data<int64_t>();
    data[0] = this->src.rows;
    data[1] = this->src.cols;
    this->infer_request.set_tensor(this->names.original_images_size.name,
                                   original_size);
  }

  this->infer_request.infer();

  auto output = this->infer_request.get_tensor(this->names.output_masks.name);
  auto shape = output.get_shape();
  OVS_ASSERT(shape.size() >= 4, "unsupported shape output mask");

  int channel = int(shape[1]);
  int height = int(shape[2]);
  int width = int(shape[3]);
  if (shape.size() == 5) {
    channel = int(shape[2]);
    height = int(shape[3]);
    width = int(shape[4]);
  }

  // NCHW
  int dims[] = {1, channel, height, width};
  const auto src = output.data();
  cv::Mat mask(4, dims, this->names.output_masks.cv_depth, src);

  std::vector<cv::Mat> interleave;
  cv::dnn::imagesFromBlob(mask, interleave);
  interleave[0].convertTo(interleave[0], CV_MAKETYPE(CV_8U, channel), 255);

  cv::Mat gray;
  if (interleave[0].type() != CV_8UC1)
    cv::cvtColor(interleave[0], gray, cv::COLOR_BGR2GRAY);
  else
    gray = interleave[0];
  cv::threshold(gray, gray, 30, 255, cv::THRESH_BINARY);

  // padding with black addressing cut off object
  cv::copyMakeBorder(gray, gray, 5, 5, 5, 5, cv::BorderTypes::BORDER_CONSTANT);

  auto edge_detection = classic::sobel(3, 3);
  auto edge = edge_detection(gray);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(edge, contours, cv::noArray(), cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  for (const auto contour : contours) {
    auto area = cv::contourArea(contour);
    // need dynamic?
    if (area < 50)
      continue;
    found(this->src, contour, 0);
  }
}

} // namespace segmentation::image::deeplearning::sam
