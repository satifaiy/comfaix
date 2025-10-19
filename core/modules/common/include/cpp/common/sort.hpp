#pragma once

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <vector>

// @brief a common data type
namespace comfaix::common {

// @brief a function that provide value that belong to the key
template <typename K, typename V>
using CacheSortValue = std::function<V(const K &)>;

// @brief a utility class provide wrapper cache for sort vector where value
// need calculation.
template <typename K, typename V> class CacheVectorSort {
public:
  CacheVectorSort(CacheSortValue<K, V> valuer) noexcept : valuer(valuer) {}

  V at(const K &key) {
    auto ptr = const_cast<K *>(&key);
    if (!this->cache.contains(ptr)) {
      this->cache[ptr] = this->valuer(key);
    }
    return this->cache[ptr];
  }

  void clear() { this->cache.clear(); }

  std::vector<K>::iterator lower_bound(std::vector<K> &vec, const K &element) {
    return std::lower_bound(vec.begin(), vec.end(), element,
                            [this](const K &a, const K &b) -> bool {
                              return this->at(a) < this->at(b);
                            });
  }

  void sort(std::vector<K> &vec) { this->sort(vec.begin(), vec.end()); }

  void sort(std::vector<K>::iterator start, std::vector<K>::iterator end) {
    std::sort(start, end,
              [this](K &a, K &b) -> bool { return this->at(a) < this->at(b); });
  }

private:
  CacheSortValue<K, V> valuer;
  std::unordered_map<K *, V> cache;
};

} // namespace comfaix::common
