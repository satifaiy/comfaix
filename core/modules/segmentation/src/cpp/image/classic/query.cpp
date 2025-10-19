#include <algorithm>
#include <cfloat>
#include <clipper2/clipper.h>
#include <cmath>
#include <cstdint>
#include <functional>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <sstream>
#include <variant>
#include <vector>

#include "common/ovs_assert.hpp"
#include "segmentation/image/classic/detection.hpp"
#include "segmentation/image/classic/query.hpp"

namespace clip = Clipper2Lib;

namespace segmentation::image::classic {

// @brief build query for classic segmentation
Queries Detection::build_queries(const std::vector<cv::Mat> &src,
                                 const std::vector<double> &min_scores,
                                 const uint32_t tolerance) {
  OVS_ASSERT(min_scores.size() == 1 || min_scores.size() == src.size(),
             "minimum score size does not match src size");

  std::ostringstream ss;

  auto queries =
      Queries{{}, DBL_MAX, 0, cv::Size(INT_MAX, INT_MAX), cv::Size(0, 0)};
  std::optional<std::reference_wrapper<Queries>> qq = std::ref(queries);
  for (int i = 0; i < src.size(); i++) {
    auto q = src[i];
    auto qc = this->edge_detection(q);

    if (qc.type() != CV_8UC1)
      cv::cvtColor(qc, qc, cv::COLOR_BGR2GRAY);
    cv::threshold(qc, qc, 128, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> query_contours;
    cv::findContours(qc, query_contours, cv::RETR_LIST,
                     cv::CHAIN_APPROX_SIMPLE);

    clip::Paths64 paths;
    for (const auto q : query_contours) {
      clip::Path64 path;
      for (const auto p : q) {
        path << clip::Point64(p.x, p.y);
      }
      paths << path;
    }

    auto union_paths =
        clip::Union(paths, clip::Paths64(), clip::FillRule::NonZero);
    auto largest =
        std::max_element(union_paths.begin(), union_paths.end(),
                         [](const clip::Path64 &a, const clip::Path64 &b) {
                           return std::fabs(Area(a)) < std::fabs(Area(b));
                         });

    std::vector<cv::Point> final_cv_contours;
    for (const auto &pt : *largest) {
      // Cast the 64-bit integer coordinates back to OpenCV's 32-bit integers.
      final_cv_contours.emplace_back(cv::Point((int)pt.x, (int)pt.y));
    }

    // crop, todo: using segment_util from pipeline instead.
    cv::Mat cropped;
    if (!crop(q, cropped, final_cv_contours))
      continue;

    double score = min_scores[0];
    if (min_scores.size() > 1) {
      score = min_scores[i];
    }

    ClassicAttribute attr = {};
    std::optional<std::reference_wrapper<ClassicAttribute>> opt_attr =
        std::ref(attr);
    auto shape = shape_attr(final_cv_contours, tolerance, opt_attr, qq);
    if (shape != ShapeAttr::invalid) {
      queries.data.emplace_back(ClassicQuery{
          cropped,
          attr,
          score,
      });
    }
  }
  return queries;
}

// @brief shape of the points with need extra compute data
ShapeAttr Detection::shape_attr(const std::vector<cv::Point> &points,
                                const uint32_t &tolerance) {
  std::optional<std::reference_wrapper<ClassicAttribute>> attr = std::nullopt;
  std::optional<std::reference_wrapper<Queries>> queries = std::nullopt;
  return shape_attr(points, tolerance, attr, queries);
}

// @brief contour classification
ShapeAttr Detection::shape_attr(
    const std::vector<cv::Point> &points, const uint32_t &tolerance,
    std::optional<std::reference_wrapper<ClassicAttribute>> &attr,
    std::optional<std::reference_wrapper<Queries>> &queries) {
  if (points.size() < 3) {
    return ShapeAttr::invalid;
  }

  double perimeter = cv::arcLength(points, true);
  double area = 0;
  if (attr.has_value())
    area = attr->get().geometry.area;
  if (area == 0)
    area = cv::contourArea(points);

  if (area < 20) {
    // to small
    return ShapeAttr::invalid;
  }

  // ******** check if contour is like a rectangle ********
  double epsilon = 0.04 * perimeter;
  std::vector<cv::Point> approx;
  cv::approxPolyDP(points, approx, epsilon, true);
  int num_vertices = approx.size();
  if (num_vertices < 3) {
    return ShapeAttr::invalid;
  }

  cv::Rect bounding_box = cv::boundingRect(points);

  if (attr.has_value()) {
    auto &atr = attr->get();
    atr.geometry.area = area;
    // this probably not a good idea
    if (tolerance > 0) {
      // use inflate and deflate to calculate max and min area
      clip::Path64 path;
      for (const auto p : points) {
        path << clip::Point64(p.x, p.y);
      }
      if (path.front() != path.back()) {
        path.push_back(path.front());
      }

      clip::Paths64 paths;
      paths << path;
      auto inflate = clip::InflatePaths(paths, tolerance, clip::JoinType::Round,
                                        clip::EndType::Polygon);

      atr.geometry.max_area = clip::Area(inflate);
      atr.geometry.min_area = area - (atr.geometry.max_area - area);
      if (atr.geometry.min_area <= 0) {
        // use percentage as fallback for too small object
        atr.geometry.min_area = area * 0.8;
      }
    } else {
      atr.geometry.min_area = 0;
      atr.geometry.max_area = DBL_MAX;
    }
  }

  if (queries.has_value()) {
    // max - min area
    auto &q = queries->get();
    if (tolerance > 0) {
      auto &atr = attr->get();
      q.max_area = std::max(q.max_area, atr.geometry.max_area);
      q.min_area = std::min(q.min_area, atr.geometry.min_area);
    }

    // max - min boundinb box size
    int max_box_width = bounding_box.width + tolerance;
    int max_box_height = bounding_box.height + tolerance;
    int min_box_width = bounding_box.width - tolerance;
    int min_box_height = bounding_box.height - tolerance;

    q.max_size.width = std::max(q.max_size.width, max_box_width);
    q.max_size.height = std::max(q.max_size.height, max_box_height);
    q.min_size.width = std::min(q.min_size.width, min_box_width);
    q.min_size.height = std::min(q.min_size.height, min_box_height);
  }

  // 4 vertices and the surface or area of contour and it's bounding box
  // should be relative close if completely filled.
  if (num_vertices == 4) {
    double box_area = (double)bounding_box.width * bounding_box.height;
    if (std::abs(area / box_area) > 0.95) {
      if (attr.has_value()) {
        auto &atr = attr->get();
        atr.shape = ShapeAttr::rectangle;
        atr.tolerance = tolerance;
        atr.polygon = points;
        if (tolerance > 0) {
          atr.geometry.min_size = cv::Size(bounding_box.width - tolerance,
                                           bounding_box.height - tolerance);
          atr.geometry.max_size = cv::Size(bounding_box.width + tolerance,
                                           bounding_box.height + tolerance);
        } else {
          atr.geometry.min_size = {0, 0};
          atr.geometry.max_size = {INT_MAX, INT_MAX};
        }
        atr.geometry.known_shape = Rectangle{bounding_box};
        atr.geometry.ratio =
            (float)std::min(bounding_box.width, bounding_box.height) /
            std::max(bounding_box.width, bounding_box.height);
      }
      return ShapeAttr::rectangle;
    }
  }

  // ******** check if contour is a circle or ellipse ********
  cv::Point2f center;
  float radius;
  cv::minEnclosingCircle(points, center, radius);
  double circle_area = M_PI * radius * radius;
  // bounding box for circle is a square
  double box_ratio = (double)std::min(bounding_box.width, bounding_box.height) /
                     std::max(bounding_box.width, bounding_box.height);

  if (area / circle_area > 0.90 && box_ratio > 0.90) {
    auto &atr = attr->get();
    atr.shape = ShapeAttr::circle;
    atr.polygon = points;
    atr.tolerance = tolerance;
    if (tolerance > 0) {
      atr.geometry.known_shape =
          Circle{radius - tolerance, radius + tolerance, radius};
    } else {
      atr.geometry.known_shape = Circle{0, FLT_MAX, radius};
    }
    return ShapeAttr::circle;
  }

  // ******** check of ellipse ********
  if (points.size() >= 5) {
    cv::RotatedRect ellipse = cv::fitEllipse(points);
    double area = M_PI * ellipse.size.width * ellipse.size.height / 4.0;
    // axis ration, minor is less than major and it ratio should be less than
    // threshold where we consider it as circle
    double major_minor_ratio =
        std::min(ellipse.size.width, ellipse.size.height) /
        std::max(ellipse.size.width, ellipse.size.height);
    // majorMinorAxisRatio should be less than 0.90 depend on threshold of
    // circle about.
    if (area / area > 0.65 && major_minor_ratio <= 0.90) {
      if (attr.has_value()) {
        auto &atr = attr->get();
        atr.shape = ShapeAttr::ellipse;
        atr.polygon = points;
        atr.tolerance = tolerance;
        if (tolerance > 0) {
          atr.geometry.known_shape = Ellipse{
              ellipse.size.height - tolerance,
              ellipse.size.height + tolerance,
              ellipse.size.width - tolerance,
              ellipse.size.width + tolerance,
              ellipse.size.height,
              ellipse.size.width,
          };
        } else {
          atr.geometry.known_shape = Ellipse{
              0, FLT_MAX, 0, FLT_MAX, ellipse.size.height, ellipse.size.width,
          };
        }
        atr.geometry.ratio = major_minor_ratio;
      }
      return ShapeAttr::ellipse;
    }
  }

  if (attr.has_value()) {
    auto &atr = attr->get();
    atr.shape = ShapeAttr::invalid;
    atr.polygon = points;
    atr.tolerance = tolerance;
  }
  switch (num_vertices) {
  case 3:
    if (attr.has_value())
      attr->get().shape = ShapeAttr::triangle;
    return ShapeAttr::triangle;
  case 4:
    // not fit rectangle, it could be trapezoid or ther shape with 4 vertices
    if (attr.has_value())
      attr->get().shape = ShapeAttr::quadrilateral;
    return ShapeAttr::quadrilateral;
  case 5:
    if (attr.has_value())
      attr->get().shape = ShapeAttr::pentagon;
    return ShapeAttr::pentagon;
  case 6:
    if (attr.has_value())
      attr->get().shape = ShapeAttr::hexagon;
    return ShapeAttr::hexagon;
  case 7:
    if (attr.has_value())
      attr->get().shape = ShapeAttr::heptagon;
    return ShapeAttr::heptagon;
  case 8:
    if (attr.has_value())
      attr->get().shape = ShapeAttr::octagon;
    return ShapeAttr::octagon;
  default:
    if (attr.has_value())
      attr->get().shape = ShapeAttr(num_vertices);
    return ShapeAttr(num_vertices);
  }
}

// @brief check if contour is similar
bool ClassicAttribute::similar(const ClassicAttribute &other) const {
  double score = cv::matchShapes(other.polygon, this->polygon,
                                 cv::ShapeMatchModes::CONTOURS_MATCH_I1, 0);
  score = std::ceil(score * 100) / 100;
  if (score < 0.1) {
    if (this->shape == other.shape) {
      // same shape
      return other.geometry.within(this->shape, this->geometry);
    }
  }
  return false;
}

// get rectangle from known_shape
const Rectangle *Geometry::rectangle() const {
  return std::get_if<Rectangle>(&this->known_shape);
}

// get circle from known_shape
const Circle *Geometry::circle() const {
  return std::get_if<Circle>(&this->known_shape);
}

// get ellipse from known_shape
const Ellipse *Geometry::ellipse() const {
  return std::get_if<Ellipse>(&this->known_shape);
}

// performe check on value geometry
bool Geometry::within(const ShapeAttr &shape, const Geometry &other) const {
  // same shape
  switch (shape) {
  case ShapeAttr::circle: {
    const Circle *circle = this->circle();
    const Circle *ocircle = other.circle();
    return circle->min_radius <= ocircle->radius &&
           ocircle->radius <= circle->max_radius;
  }
  case ShapeAttr::ellipse: {
    const Ellipse *ellipse = this->ellipse();
    const Ellipse *oellipse = other.ellipse();
    return ellipse->min_major <= oellipse->major &&
           oellipse->major <= ellipse->max_major &&
           ellipse->min_minor <= oellipse->minor &&
           oellipse->minor <= ellipse->max_minor;
  }
  case ShapeAttr::rectangle: {
    const Rectangle *rect = this->rectangle();
    const Rectangle *orect = other.rectangle();
    return this->min_size.width <= orect->bounding_box.width &&
           orect->bounding_box.width <= this->max_size.width &&
           this->min_size.height <= orect->bounding_box.height &&
           orect->bounding_box.height <= this->max_size.height;
  }
  default:
    return true;
  }
}

// @brief crop the source image
// @return true if contour points produce minimum area to be crop otherwise
// false.
bool crop(const cv::Mat &source, cv::Mat &out,
          const std::vector<cv::Point> points) {
  cv::Rect bbox = cv::boundingRect(points);
  bbox.x = std::max(0, bbox.x);
  bbox.y = std::max(0, bbox.y);
  bbox.width = std::min(source.cols - bbox.x, bbox.width);
  bbox.height = std::min(source.rows - bbox.y, bbox.height);
  if (bbox.width <= 0 || bbox.height <= 0) {
    return false;
  }

  cv::Mat mask = cv::Mat::zeros(source.size(), CV_8UC1);
  cv::fillPoly(mask, points, cv::Scalar(255));
  cv::Mat masked;
  source.copyTo(masked, mask);

  cv::Mat cropped = masked(bbox);
  // need rotate??

  cv::Mat gray;
  if (cropped.channels() == 3) {
    cv::cvtColor(cropped, gray, cv::COLOR_BGR2GRAY);
  } else if (cropped.channels() == 4) {
    cv::cvtColor(cropped, gray, cv::COLOR_BGRA2GRAY);
  } else {
    gray = cropped.clone();
  }

  // threshold/boolean mask (single-channel CV_8UC1)
  cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY);

  if (cv::countNonZero(gray) != 0) {
    cv::Rect tight = cv::boundingRect(gray);
    // guard bounds just in case
    tight.x = std::max(0, tight.x);
    tight.y = std::max(0, tight.y);
    tight &= cv::Rect(0, 0, cropped.cols, cropped.rows);
    out = cropped(tight).clone();
  } else {
    out = cropped.clone();
  }
  return true;
}

} // namespace segmentation::image::classic
