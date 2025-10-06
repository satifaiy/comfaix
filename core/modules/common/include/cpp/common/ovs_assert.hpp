#ifndef OPENVINO_WRAPPER_ASSERT_HPP
#define OPENVINO_WRAPPER_ASSERT_HPP

#include <sstream>
#include <stdexcept>
#include <string>

namespace comfaix::common {

static inline std::ostream &write_all_to_stream(std::ostream &str) {
  return str;
}

template <typename T, typename... TS>
std::ostream &write_all_to_stream(std::ostream &str, T &&arg, TS &&...args) {
  return write_all_to_stream(str << arg, std::forward<TS>(args)...);
}

// A custom exception class for assertions.
class AssertionError : public std::runtime_error {
public:
  explicit AssertionError(const std::string &message)
      : std::runtime_error(message) {}
};

} // namespace comfaix::common

#define OVS_ASSERT(expr, ...)                                                  \
  do {                                                                         \
    if (!static_cast<bool>(expr)) {                                            \
      ::std::ostringstream oss;                                                \
      oss << "Assertion Failed! " << __FILE__ << ":" << __LINE__ << "\n"       \
          << "  Expression: " << #expr << "\n"                                 \
          << "  Message: ";                                                    \
      ::comfaix::common::write_all_to_stream(oss, __VA_ARGS__);                \
      throw comfaix::common::AssertionError(oss.str());                        \
    }                                                                          \
  } while (0)

#endif // OPENVINO_WRAPPER_ASSERT_HPP
