#include <algorithm>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/runtime/core.hpp>
#include <vector>

#include "pipeline/image/normalize_op.hpp"
#include "pipeline/image/permute_op.hpp"
#include "pipeline/image/resize_op.hpp"
#include "pipeline/image/segment_op.hpp"
#include "segmentation/text/paddle/detect.hpp"

// shorten the namespace
namespace pimg = pipeline::image::op;

namespace segmentation::text::paddle {

// @brief create Detect with the given model path and device.
// @param path is the path to the text detection model.
// @param device is the openvino backend
Detection::Detection(std::string path, std::string device)
    : Detection(read_detect_config(path, device)) {}

// @brief create Detect with the given model config and device.
// @param detect model configuration
// @param device is the openvino backend
Detection::Detection(const DetectConfig &config) : config(config) {
  ov::Core core;
  this->model = core.read_model(config.model_path);
  // std::shared_ptr<ov::Model> model;
  this->compiled_model = core.compile_model(this->model, config.device);
  this->infer_request = this->compiled_model.create_infer_request();
}

// Detection::~Detection() {}

// @brief implement ISegementation.
std::vector<common::ImageSegmentionResult>
Detection::segment(const cv::Mat &m) {
  cv::Size2f ratio;
  const auto input_tensor =
      detect_pre_processing(m, this->compiled_model, ratio, this->config);
  this->infer_request.set_input_tensor(input_tensor);
  this->infer_request.infer();

  // read only tensor and prediction as well as bitmap result
  auto output = this->infer_request.get_output_tensor(0);
  const float *out_data = output.data<const float>();

  ov::Shape output_shape = output.get_shape();
  const size_t n2 = output_shape[2];
  const size_t n3 = output_shape[3];
  const int n = n2 * n3;

  std::vector<float> pred(n, 0.0);
  std::vector<unsigned char> cbuf(n, ' ');

  for (int i = 0; i < n; i++) {
    pred[i] = float(out_data[i]);
    cbuf[i] = (unsigned char)((out_data[i]) * 255);
  }

  cv::Mat bitmap(n2, n3, CV_8UC1, (unsigned char *)cbuf.data());
  cv::Mat prediction(n2, n3, CV_32F, (float *)pred.data());

  // segment the image
  auto align = [](cv::Mat &src, const std::vector<cv::Point2f> &points,
                  const cv::RotatedRect &rotated) -> cv::Mat {
    float angle = rotated.angle;
    auto size = src.size();
    auto center = rotated.center;
    if (angle > 10)
      angle -= 90;

    cv::Mat m;
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
    double cos_theta = rot_mat.at<double>(0, 0);
    double sin_theta = rot_mat.at<double>(0, 1);
    int wn = (int)std::round(std::abs(src.cols * cos_theta) +
                             std::abs(src.rows * sin_theta));
    int hn = (int)std::round(std::abs(src.cols * sin_theta) +
                             std::abs(src.rows * cos_theta));

    rot_mat.at<double>(0, 2) += (wn / 2.0f) - center.x;
    rot_mat.at<double>(1, 2) += (hn / 2.0f) - center.y;
    cv::warpAffine(src, m, rot_mat, cv::Size(wn, hn), cv::INTER_CUBIC);
    return m;
  };
  pimg::Cropper copper = pimg::create_cropper(m, align);
  auto result = pimg::db_segmentation(bitmap, copper, this->config.db_config,
                                      m.size(), ratio);
  return result;
}

// @brief pre-processing step for paddle text detection
ov::Tensor Detection::detect_pre_processing(const cv::Mat &m,
                                            ov::CompiledModel compiled_model,
                                            cv::Size2f &ratio,
                                            const DetectConfig &config) {
  // resize to N-Multiple Block Resized size
  auto resized =
      pimg::resize_n_mutiply(m, config.max_width_height, config.block_size,
                             ratio, cv::InterpolationFlags::INTER_LINEAR);
  // normalize value to 0-1 float
  resized = pimg::min_max_normalize(resized, CV_32FC3);
  // normalize value with z-score
  resized = pimg::standard_score_normalize(resized, CV_32FC1, config.means,
                                           config.scales);
  // permute HWC t= CHW
  auto input_port = compiled_model.input();
  ov::Shape intput_shape = {1, 3, static_cast<size_t>(resized.rows),
                            static_cast<size_t>(resized.cols)};
  ov::Tensor input_tensor(input_port.get_element_type(), intput_shape);
  pimg::permute(resized, input_tensor.data<float>());
  return input_tensor;
}

// @brief provide simple consistent sort order.
// @detail the sort is group by rows and then by columns. If the segment is a
// row span or column span the segment will add to earliest row or columns.
// To have procise location, use segmentation::GridVisualization which layout
// the exact location of the detected text on the image. It's also providing
// support for both column and row span.
void Detection::sort_results(
    std::vector<common::ImageSegmentionResult> &results, float tolerance) {
  std::unordered_map<common::ImageSegmentionResult *, cv::Rect> cache_boxes;
  std::sort(results.begin(), results.end(),
            [&cache_boxes,
             tolerance](const common::ImageSegmentionResult &a,
                        const common::ImageSegmentionResult &b) -> bool {
              auto aptr = const_cast<common::ImageSegmentionResult *>(&a);
              auto bptr = const_cast<common::ImageSegmentionResult *>(&b);
              cv::Rect box1, box2;
              if (cache_boxes.contains(aptr)) {
                box1 = cache_boxes[aptr];
              } else {
                box1 = cv::boundingRect(a.roi);
                cache_boxes[aptr] = box1;
              }
              if (cache_boxes.contains(bptr)) {
                box2 = cache_boxes[bptr];
              } else {
                box2 = cv::boundingRect(b.roi);
                cache_boxes[bptr] = box2;
              }
              return compare_result(box1, box2, tolerance);
            });
}

// &brief compare 2 rectangle
bool Detection::compare_result(const cv::Rect &a, const cv::Rect &b,
                               float tolerance) {
  // check if vertically overlap
  bool overlap_y =
      // "a" y-axis range is within "b" y-axis range
      ((a.y + tolerance) >= (b.y - tolerance) &&
       ((a.y + a.height - tolerance) <= (b.y + b.height + tolerance))) ||
      // "b" y-axis range is within "a" y-axis range
      ((b.y + tolerance) >= (a.y - tolerance) &&
       ((b.y + b.height - tolerance) <= (a.y + a.height + tolerance)));

  // if not overlap y then this is different row
  if (!overlap_y) {
    return a.y < b.y;
  }
  return a.x < b.x;
}

} // namespace segmentation::text::paddle
