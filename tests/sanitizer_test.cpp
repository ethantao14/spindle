// Compiled only in the sanitizer build. Proves the sanitizers are really on,
// so a misconfigured build cannot pass the suite while catching nothing.
#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <vector>

namespace {

// Reading through volatile stops the compiler from proving the bugs at compile time.
int read_past_end(std::size_t index) {
  std::vector<int> values(4, 0);
  const volatile std::size_t opaque_index = index;
  // Volatile forces the read; a raw pointer avoids a library bounds check.
  const volatile int* raw_values = values.data();
  return raw_values[opaque_index];
}

int overflow_signed(int value) {
  const volatile int opaque_value = value;
  return opaque_value + 1;
}

}  // namespace

TEST(SanitizerDeathTest, AddressSanitizerCatchesHeapOverflow) {
  EXPECT_DEATH(read_past_end(4), "heap-buffer-overflow");
}

TEST(SanitizerDeathTest, UndefinedBehaviorSanitizerCatchesSignedOverflow) {
  EXPECT_DEATH(overflow_signed(std::numeric_limits<int>::max()), "signed integer overflow");
}
