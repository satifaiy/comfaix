#pragma one

#include <opencv2/core/mat.hpp>
#include <optional>
#include <vector>

#include "common/ovs_assert.hpp"

namespace pipeline {

// @brief represent the source to be used as batch processing
template <typename G> struct Source {
  const cv::Mat &origin;
  std::vector<G> data;
};

// @brief the interface represent step processing
template <typename T, typename G, typename R> class IProcessing {
public:
  virtual ~IProcessing() = default;

  // @brief initialize process processing step. This function will be call when
  // a new batch started.
  virtual T batch_initialize(int start, int count,
                             const std::vector<G> &segments) = 0;

  // @brief handling each processing item
  virtual G batch_processing(int index, const cv::Mat &origin, T &t,
                             G &segment) = 0;

  // @brief call after all items has been processed.
  virtual R batch_post_processing(T &t, int start, int count,
                                  const std::vector<G> &segments) = 0;
};

// @brief iterator provide handling batch processing
template <typename T, typename G, typename R> class BatchIterator {
private:
  const Source<G> &raw_data;
  const int batch_size;

  int current_index;
  std::optional<R> cached_batch;

  IProcessing<T, G, R> *processing;

public:
  using iterator_category = std::input_iterator_tag;
  using difference_type = std::ptrdiff_t;

  // Constructor for the 'begin' iterator
  BatchIterator(const Source<G> &data, size_t size, size_t index,
                IProcessing<T, G, R> *preprocessing)
      : raw_data(data), batch_size(size), current_index(index),
        processing(preprocessing) {};

  // Constructor for the 'end' sentinel (index is equal to data size)
  BatchIterator(const Source<G> &data, size_t size,
                IProcessing<T, G, R> *preprocessing)
      : raw_data(data), batch_size(size), current_index(data.data.size()),
        processing(preprocessing) {};

  R &operator*() {
    if (!cached_batch.has_value()) {
      size_t items_to_process =
          std::min(batch_size, int(this->raw_data.data.size()) - current_index);

      // call the batch initialize to create necessary data
      T attr = this->processing->batch_initialize(
          current_index, items_to_process, this->raw_data.data);

      // call for processing for each item
      for (size_t i = 0; i < items_to_process; ++i) {
        G current_item = this->raw_data.data[current_index + i];
        this->processing->batch_processing(i, this->raw_data.origin, attr,
                                           current_item);
      }

      // call for post processing
      R merge = this->processing->batch_post_processing(
          attr, current_index, items_to_process, this->raw_data.data);

      cached_batch = std::move(merge);
    }
    return cached_batch.value();
  }

  BatchIterator &operator++() {
    if (this->current_index < this->raw_data.data.size()) {
      this->current_index += batch_size;
      if (this->current_index > this->raw_data.data.size()) {
        // end of loop
        this->current_index = this->raw_data.data.size();
      }
      cached_batch.reset();
    }
    return *this;
  }

  BatchIterator operator++(int) {
    BatchIterator tmp = *this;
    ++(*this);
    return tmp;
  }

  bool operator==(const BatchIterator &other) const {
    return this->current_index == other.current_index;
  }

  bool operator!=(const BatchIterator &other) const {
    return !(*this == other);
  }
};

/**
 * @brief A container class that provides 'begin' and 'end' iterators
 * to make the BatchIterator usable in a range-based for loop.
 */
template <typename T, typename G, typename R> class BatchProcessor {
private:
  size_t batch_size;
  std::unique_ptr<IProcessing<T, G, R>> processing;
  // can be set before iterator
  std::optional<const Source<G>> raw_data;

public:
  BatchProcessor(int batch_size,
                 std::unique_ptr<IProcessing<T, G, R>> preprocessing)
      : batch_size(batch_size), processing(std::move(preprocessing)) {};

  BatchProcessor(const Source<G> &data, int batch_size,
                 std::unique_ptr<IProcessing<T, G, R>> preprocessing)
      : batch_size(batch_size), processing(std::move(preprocessing)) {
    this->raw_data.reset();
    this->raw_data.emplace(data);
  };

  void set_source(const Source<G> data) {
    this->raw_data.reset();
    this->raw_data.emplace(data);
  }

  BatchIterator<T, G, R> begin() const {
    OVS_ASSERT(raw_data.has_value(), "no source data provided");
    return BatchIterator(raw_data.value(), batch_size, size_t(0),
                         processing.get());
  }

  BatchIterator<T, G, R> end() const {
    OVS_ASSERT(raw_data.has_value(), "no source data provided");
    return BatchIterator(raw_data.value(), batch_size, processing.get());
  }
};

} // namespace pipeline
