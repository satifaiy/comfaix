#include <algorithm>
#include <gtest/gtest.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/runtime/core.hpp>

#include "common/similarity.hpp"
#include "segmentation/image/deeplearning/sam.hpp"
#include "segmentation/image/deeplearning/vit.hpp"

namespace sam = segmentation::image::deeplearning::sam;
namespace vit = segmentation::image::deeplearning::vit;
namespace dino = segmentation::image::deeplearning::vit::dino;

struct TestCaseBoxImg {
  sam::Box box;
  std::string image;
};

TEST(TestSAM, TestSegment) {
  std::filesystem::path model_cache("../../../../../../models/openvino");
  std::filesystem::path model_path(
      "../../../../../../models/yunyangx/EfficientSAM/efficientsam_s.onnx");
  std::filesystem::path vit_path("../../../../../../models/onnx-community/"
                                 "dinov2-small/onnx/model_quantized.onnx");

  const auto config = dino::v2::read_config(
      vit_path.parent_path().parent_path().append("preprocessor_config.json"));
  vit::ViT vision(vit_path, "AUTO", model_cache,
                  dino::v2::create_preprocessing(config));

  sam::SAM model_sam(
      model_path.parent_path().append("efficientsam_ti_encoder.onnx"),
      model_path.parent_path().append("efficientsam_ti_decoder.onnx"), "AUTO",
      model_cache);
  auto source = cv::imread("data/image/deeplearning/game.jpeg");
  auto infer = model_sam.create_infer(source);

  std::vector<TestCaseBoxImg> tests = {
      {{389, 26, 389 + 42, 26 + 53},
       "data/image/deeplearning/game_health_portion.png"},
      {{208, 31, 208 + 49, 31 + 46}, "data/image/deeplearning/game_diamon.png"},
      {{14, 20, 14 + 68, 20 + 56},
       "data/image/deeplearning/game_gold_coin.png"},
      {{18, 84, 18 + 56, 86 + 56},
       "data/image/deeplearning/game_energy_portion.png"},
      {{616, 870, 616 + 30, 870 + 30},
       "data/image/deeplearning/game_reddot.png"}};

  for (const auto in : tests) {
    auto results = infer.segment({}, {in.box});
    ASSERT_EQ(results.size(), 1);

    cv::Mat img = cv::imread(in.image);
    cv::Mat exp;
    if (img.channels() == 4) {
      std::vector<cv::Mat> channels;
      cv::split(img, channels);
      cv::Mat alpha_channel = channels[3];
      cv::Mat non_zero_locs;
      cv::findNonZero(alpha_channel, non_zero_locs);
      cv::Rect bounding_box;
      bounding_box = cv::boundingRect(non_zero_locs);
      exp = img(bounding_box);
    } else {
      exp = img;
    }

    cv::Mat seg;
    if (exp.rows != results[0].segment.rows ||
        exp.cols != results[0].segment.cols) {
      cv::Size size((exp.cols + results[0].segment.cols) / 2,
                    (exp.rows + results[0].segment.rows) / 2);
      cv::resize(results[0].segment, seg, size);
      cv::resize(exp.clone(), exp, size);
    } else {
      seg = results[0].segment;
    }

    auto a = vision.embed(exp);
    auto b = vision.embed(seg);
    auto score = common::cosine_similarity<float>(
        dino::v2::global_token<float>(a), dino::v2::global_token<float>(b));
    ASSERT_GT(score, 0.90);
  }

  source = cv::imread("data/image/deeplearning/tiger.png");
  auto infer_tiger = model_sam.create_infer(source);
  auto results = infer_tiger.segment({}, {{300, 130, 1426, 920}});
  auto which = std::max_element(
      results.begin(), results.end(),
      [](const sam::Result &a, const sam::Result &b) -> bool {
        return a.segment.size().area() < b.segment.size().area();
      });
  auto b = vision.embed(which->segment);
  auto tiger_embed = dino::v2::global_token<float>(b);

  auto exact = cv::imread("data/image/deeplearning/tiger_exact.png");
  auto a = vision.embed(exact);
  auto score = common::cosine_similarity<float>(
      dino::v2::global_token<float>(a), tiger_embed);
  ASSERT_GT(score, 0.99);

  auto similar = cv::imread("data/image/deeplearning/tiger_similar.png");
  a = vision.embed(similar);
  score = common::cosine_similarity<float>(dino::v2::global_token<float>(a),
                                           tiger_embed);
  ASSERT_GT(score, 0.85);

  similar = cv::imread("data/image/deeplearning/aussie.png");
  auto aussie = vision.embed(similar);
  auto aussie_embed = dino::v2::global_token<float>(a);
  score = common::cosine_similarity<float>(aussie_embed, tiger_embed);
  ASSERT_LT(score, 0.1);

  source = cv::imread("data/image/deeplearning/dog_cat.jpg");
  auto infer_dog_cat = model_sam.create_infer(source);
  results = infer_dog_cat.segment({}, {{90, 130, 90 + 440, 130 + 200}});
  ASSERT_GT(results.size(), 0);
  which = std::max_element(
      results.begin(), results.end(),
      [](const sam::Result &a, const sam::Result &b) -> bool {
        return a.segment.size().area() < b.segment.size().area();
      });
  auto dog_vector = vision.embed(which->segment);
  auto dog_embed = dino::v2::global_token<float>(dog_vector);

  score = common::cosine_similarity<float>(aussie_embed, dog_embed);
  ASSERT_GT(score, 0.60);
}
