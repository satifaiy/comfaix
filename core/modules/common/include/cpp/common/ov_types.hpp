#pragma once

#include <cstdint>
#include <openvino/core/type/element_type.hpp>
#include <openvino/core/type/float16.hpp>

// @brief a common data type
namespace comfaix::common {

template <typename Function>
auto dispatch_by_type(ov::element::Type t, Function &&func) {
  switch (t) {
  case ov::element::f32:
    return func.template operator()<float>();
  case ov::element::f64:
    return func.template operator()<double>();
  case ov::element::i8:
    return func.template operator()<int8_t>();
  case ov::element::i16:
    return func.template operator()<int16_t>();
  case ov::element::i32:
    return func.template operator()<int32_t>();
  case ov::element::i64:
    return func.template operator()<int64_t>();
  case ov::element::u8:
    return func.template operator()<uint8_t>();
  case ov::element::u16:
    return func.template operator()<uint16_t>();
  case ov::element::u32:
    return func.template operator()<uint32_t>();
  case ov::element::u64:
    return func.template operator()<uint64_t>();
  default:
    throw std::runtime_error("Unsupported element type.");
  }
}

} // namespace comfaix::common
