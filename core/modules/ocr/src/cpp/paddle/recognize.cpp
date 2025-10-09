#include <algorithm>
#include <openvino/runtime/core.hpp>
#include <string>

#include "common/segmentation.hpp"
#include "ocr/paddle/recognize.hpp"
#include "segmentation/text/paddle/detect.hpp"

namespace ocr::paddle {

// @brief create Recognize with the given model path and device.
// @param path is the path to the text detection model.
// @param device is the openvino backend
Recognize::Recognize(std::string path,
                     std::unique_ptr<Segmentation> segmentation,
                     std::string device)
    : Recognize(read_recognize_config(path, device), std::move(segmentation)) {}

// @brief create Recognize with the given model config and device.
// @param detect model configuration
// @param device is the openvino backend
Recognize::Recognize(const RecognizeConfig &config,
                     std::unique_ptr<Segmentation> segmentation)
    : config(config), segmentation(std::move(segmentation)),
      batch(config.batch_number,
            std::make_unique<RecognizePreProcessing>(config)) {
  ov::Core core;
  this->model = core.read_model(config.model_path);
  // std::shared_ptr<ov::Model> model;
  this->compiled_model = core.compile_model(this->model, config.device);
  this->infer_request = this->compiled_model.create_infer_request();
}

// @brief implement ISegementation.
std::vector<common::TextRecognitionResult>
Recognize::recognize(const cv::Mat &m) {
  auto detected = this->segmentation.get()->segment(m);
  segmentation::text::paddle::Detection::sort_results(detected);
  this->batch.set_source(
      pipeline::Source<common::ImageSegmentionResult>(m, detected));

  std::vector<std::string> rec_texts(detected.size(), "");
  std::vector<float> rec_text_scores(detected.size(), 0);
  int beg_img_no = 0;

  for (const auto tensor : this->batch) {
    this->infer_request.set_input_tensor(tensor);
    this->infer_request.infer();

    auto output = this->infer_request.get_output_tensor();
    const float *out_data = output.data<const float>();
    auto predict_shape = output.get_shape();

    // predict_batch is the result of Last FC with softmax
    for (int m = 0; m < predict_shape[0]; m++) {
      std::string str_res;
      int argmax_idx;
      int last_index = 0;
      float score = 0.f;
      int count = 0;
      float max_value = 0.0f;

      for (int n = 0; n < predict_shape[1]; n++) {
        // get idx
        argmax_idx = int(Recognize::max_score_index(
            &out_data[(m * predict_shape[1] + n) * predict_shape[2]],
            &out_data[(m * predict_shape[1] + n + 1) * predict_shape[2]]));
        // get score
        max_value = float(*std::max_element(
            &out_data[(m * predict_shape[1] + n) * predict_shape[2]],
            &out_data[(m * predict_shape[1] + n + 1) * predict_shape[2]]));

        if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
          score += max_value;
          count += 1;
          if (argmax_idx < this->config.recognized_characters.size()) {
            str_res += this->config.recognized_characters[argmax_idx];
          }
        }
        last_index = argmax_idx;
      }
      score /= count;
      if (std::isnan(score)) {
        continue;
      }
      rec_texts[beg_img_no + m] = str_res;
      rec_text_scores[beg_img_no] = score;
    }
    beg_img_no += this->config.batch_number;
  }
  std::vector<common::TextRecognitionResult> results;
  for (int i = 0; i < rec_texts.size(); i++) {
    results.push_back({rec_texts[i], detected[i].roi, rec_text_scores[i]});
  }
  return results;
}

} // namespace ocr::paddle
