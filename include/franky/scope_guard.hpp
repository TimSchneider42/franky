#include <functional>
#include <utility>

namespace franky {

/**
 * @brief A scope guard that executes a function when it goes out of scope.
 */
class scope_guard {
 public:
  /**
   * @brief Constructor that takes a function to execute when the guard goes out
   * of scope.
   *
   * @param f The function to execute.
   */
  explicit scope_guard(std::function<void()> f) : f_(std::move(f)) {}

  ~scope_guard() { f_(); }

 private:
  std::function<void()> f_;
};

}  // namespace franky
