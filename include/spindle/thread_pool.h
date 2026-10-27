#pragma once

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace spindle {

// Fixed set of worker threads for splitting one loop across cores. A pool of
// size 1 has no workers and runs everything on the calling thread.
class ThreadPool {
 public:
  // Throws std::invalid_argument if threads is 0.
  explicit ThreadPool(std::size_t threads);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  [[nodiscard]] std::size_t size() const { return workers_.size() + 1; }

  // Calls task(chunk, chunk_count) once for every chunk in [0, chunk_count), spread over
  // the pool (the caller runs some too), and returns when all are done. Rethrows the
  // first exception a task threw. Not reentrant: a task must not call run on this pool.
  void run(std::size_t chunk_count, const std::function<void(std::size_t, std::size_t)>& task);

 private:
  void work();
  void execute();

  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::condition_variable done_;
  const std::function<void(std::size_t, std::size_t)>* task_ = nullptr;
  std::size_t chunk_count_ = 0;
  std::size_t next_chunk_ = 0;
  std::size_t finished_chunks_ = 0;
  std::size_t generation_ = 0;
  bool stopping_ = false;
  std::exception_ptr error_;
};

}  // namespace spindle
