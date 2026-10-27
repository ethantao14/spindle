#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "spindle/thread_pool.h"

namespace {

TEST(ThreadPoolSpec, SizeAndZeroRejection) {
  EXPECT_THROW((void)spindle::ThreadPool(0), std::invalid_argument);
  for (const std::size_t threads : std::array<std::size_t, 5>{1, 2, 3, 4, 8}) {
    const spindle::ThreadPool pool(threads);
    EXPECT_EQ(pool.size(), threads);
  }
}

TEST(ThreadPoolSpec, EveryChunkExactlyOnceAndWaitsForCompletion) {
  for (const std::size_t threads : std::array<std::size_t, 5>{1, 2, 3, 4, 8}) {
    spindle::ThreadPool pool(threads);
    for (const std::size_t count : std::array<std::size_t, 7>{0, 1, 2, 3, 7, 64, 1000}) {
      SCOPED_TRACE(::testing::Message() << "threads=" << threads << " chunks=" << count);
      std::vector<std::atomic<unsigned>> calls(count);
      std::atomic<bool> invalid{false};
      pool.run(count, [&](std::size_t chunk, std::size_t total) {
        if (chunk >= count || total != count) {
          invalid.store(true);
          return;
        }
        calls[chunk].fetch_add(1);
      });
      EXPECT_FALSE(invalid.load());
      for (const auto& call : calls) {
        EXPECT_EQ(call.load(), 1U);
      }
    }
  }
}

TEST(ThreadPoolSpec, SizeOneUsesCallingThread) {
  spindle::ThreadPool pool(1);
  const auto caller = std::this_thread::get_id();
  std::atomic<bool> wrong_thread{false};
  pool.run(64, [&](std::size_t, std::size_t) {
    if (std::this_thread::get_id() != caller) {
      wrong_thread.store(true);
    }
  });
  EXPECT_FALSE(wrong_thread.load());
}

TEST(ThreadPoolSpec, LargerPoolUsesMultipleThreads) {
  spindle::ThreadPool pool(4);
  std::mutex mutex;
  std::set<std::thread::id> ids;
  pool.run(64, [&](std::size_t, std::size_t) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const std::scoped_lock<std::mutex> lock(mutex);
    ids.insert(std::this_thread::get_id());
  });
  EXPECT_GT(ids.size(), 1U);
}

TEST(ThreadPoolSpec, RethrowsTaskExceptionAndCanRunAgain) {
  for (const std::size_t threads : std::array<std::size_t, 5>{1, 2, 3, 4, 8}) {
    spindle::ThreadPool pool(threads);
    try {
      pool.run(7, [](std::size_t chunk, std::size_t) {
        if (chunk == 2) {
          throw std::runtime_error("chunk failure");
        }
      });
      FAIL() << "run did not rethrow";
    } catch (const std::runtime_error& error) {
      EXPECT_STREQ(error.what(), "chunk failure");
    }
    std::atomic<unsigned> sum{0};
    pool.run(7, [&](std::size_t chunk, std::size_t) {
      sum.fetch_add(static_cast<unsigned>(chunk + 1));
    });
    EXPECT_EQ(sum.load(), 28U);
  }
}

TEST(ThreadPoolSpec, BackToBackRunsDoNotLoseWakeups) {
  for (const std::size_t threads : std::array<std::size_t, 5>{1, 2, 3, 4, 8}) {
    spindle::ThreadPool pool(threads);
    std::atomic<std::size_t> sum{0};
    std::size_t expected = 0;
    for (std::size_t iteration = 0; iteration < 2000; ++iteration) {
      const auto count = iteration % 5;
      pool.run(count, [&](std::size_t chunk, std::size_t) { sum.fetch_add(chunk + 1); });
      expected += (count * (count + 1)) / 2;
      ASSERT_EQ(sum.load(), expected) << "iteration=" << iteration;
    }
  }
}

TEST(ThreadPoolSpec, DestructionAfterConstructionAndRunCompletes) {
  for (const std::size_t threads : std::array<std::size_t, 5>{1, 2, 3, 4, 8}) {
    {
      const spindle::ThreadPool pool(threads);
    }
    {
      spindle::ThreadPool pool(threads);
      pool.run(7, [](std::size_t, std::size_t) {});
    }
  }
  SUCCEED();
}

}  // namespace
