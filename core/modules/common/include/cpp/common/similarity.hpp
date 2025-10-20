#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace comfaix::common {

// @brief cosine similarity of two vector
template <typename T>
float cosine_similarity(const std::vector<T> &va, const std::vector<T> &vb) {
  if (va.size() != vb.size() || va.empty()) {
    return 0.0f;
  }

  float dot_product = 0.0f;
  float norma = 0.0f;
  float normb = 0.0f;

  for (std::size_t i = 0; i < va.size(); ++i) {
    dot_product += va[i] * vb[i];
    norma += va[i] * va[i]; // a²
    normb += vb[i] * vb[i]; // b²
  }

  float magnitude_a = std::sqrt(norma);
  float magnitude_b = std::sqrt(normb);

  if (magnitude_a == 0.0f || magnitude_b == 0.0f) {
    return 0.0f;
  }
  return dot_product / (magnitude_a * magnitude_b);
}

} // namespace comfaix::common
