#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <variant>
#include <vector>

#include "segmentation/image/query.hpp"

namespace segmentation::image::classic {

// @brief an option define how image need to resize before comparing
enum class ResizeAttr {
  None,         // no resize, it's upto similarity function
  DownSampling, // the target image is resize to fit query
  UpSampling,   // the query image is resize to fit target image
  Middle,       // both target and query resize to fit each other
};

// @brief a share of contour
enum class ShapeAttr {
  invalid = -1,
  none,
  circle,
  ellipse,
  triangle,
  rectangle,
  quadrilateral,
  pentagon,
  hexagon,
  heptagon,
  octagon
};

struct Rectangle {
  // rectangle shape attrubute
  cv::Rect bounding_box;
};

struct Circle {
  // circle shape attribute
  float min_radius, max_radius;
  float radius;
};

struct Ellipse {
  // ellipse shape attribute
  float min_minor, max_minor;
  float min_major, max_major;
  float minor, major;
};

struct Geometry {
  std::variant<Rectangle, Circle, Ellipse> known_shape;

  // general shape
  cv::Size min_size, max_size;
  double min_area, max_area;
  double area;
  double ratio; // ratio of respected type

  // get rectangle from known_shape
  const Rectangle *rectangle() const;

  // get circle from known_shape
  const Circle *circle() const;

  // get ellipse from known_shape
  const Ellipse *ellipse() const;

  // performe check on value geometry
  bool within(const ShapeAttr &shape, const Geometry &other) const;
};

// @brief a data use to detecting and comparing data
struct ClassicAttribute {
  // shape of the contour
  ShapeAttr shape;

  // path that represent the contour of an image
  std::vector<cv::Point> polygon;

  // maximum and minimum acceptable size
  Geometry geometry;

  // the size different tolerance
  size_t tolerance;

  // @brief check if contour is similar
  bool similar(const ClassicAttribute &other) const;
};

// @breif class query use for segmentation
using ClassicQuery = segmentation::image::Query<cv::Mat, ClassicAttribute>;

// @brief query data
struct Queries {
  std::vector<ClassicQuery> data;
  double min_area, max_area;
  cv::Size min_size, max_size;
};

// @brief crop the source image
// @return true if contour points produce minimum area to be crop otherwise
// false.
bool crop(const cv::Mat &source, cv::Mat &out,
          const std::vector<cv::Point> points);

} // namespace segmentation::image::classic
