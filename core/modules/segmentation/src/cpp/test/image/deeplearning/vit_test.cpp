#include <filesystem>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/quality/qualityssim.hpp>
#include <openvino/runtime/core.hpp>

#include "common/similarity.hpp"
#include "segmentation/image/deeplearning/vit.hpp"

namespace vit = segmentation::image::deeplearning::vit;
namespace dino = segmentation::image::deeplearning::vit::dino;

TEST(TestViT, TestDinoV2) {
  std::filesystem::path model_cache("../../../../../../models/openvino");
  std::filesystem::path model_path("../../../../../../models/onnx-community/"
                                   "dinov2-small/onnx/model_quantized.onnx");
  auto config =
      dino::v2::read_config(model_path.parent_path().parent_path().append(
          "preprocessor_config.json"));
  vit::ViT vision_embed(model_path, "AUTO", model_cache,
                        dino::v2::create_preprocessing(config));
  auto portion = cv::imread("data/image/deeplearning/game_energy_portion.png");
  auto output1 = vision_embed.embed(portion);
  auto fvect1 = dino::v2::global_token<float>(output1);
  ASSERT_EQ(fvect1.size(), 384);

  cv::Mat mod;

  cv::Point2f srcTri[3];
  srcTri[0] = cv::Point2f(0, 0);                // Top-left corner
  srcTri[1] = cv::Point2f(portion.cols - 1, 0); // Top-right corner
  srcTri[2] = cv::Point2f(0, portion.rows - 1); // Bottom-left corner

  cv::Point2f dstTri[3];
  dstTri[0] = cv::Point2f(0, 0);
  dstTri[1] = cv::Point2f(portion.cols - 1, 0 + 5); // Skew along Y-axis
  dstTri[2] = cv::Point2f(0 + 5, portion.rows - 1); // Skew along X-axis

  cv::Mat warpMat = cv::getAffineTransform(srcTri, dstTri);
  cv::warpAffine(portion, mod, warpMat, portion.size());

  cv::resize(portion, mod, cv::Size(portion.cols * 2, portion.rows * 2));

  auto output2 = vision_embed.embed(mod);
  auto fvect2 = dino::v2::global_token<float>(output2);
  ASSERT_EQ(fvect2.size(), 384);

  auto score = common::cosine_similarity(fvect1, fvect2);
  ASSERT_GT(score, 0.90);
}
