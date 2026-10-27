#include "spindle/thread_pool.h"

#include <stdexcept>

namespace spindle {

ThreadPool::ThreadPool(std::size_t threads) {
  if (threads == 0) {
    throw std::invalid_argument("threads must be positive");
  }
  try {
    for (std::size_t i = 1; i < threads; ++i) {
      workers_.emplace_back([this] { work(); });
    }
  } catch (...) {
    {
      const std::scoped_lock lock(mutex_);
      stopping_ = true;
    }
    wake_.notify_all();
    for (auto& worker : workers_) {
      worker.join();
    }
    throw;
  }
}

ThreadPool::~ThreadPool() {
  {
    const std::scoped_lock lock(mutex_);
    stopping_ = true;
  }
  wake_.notify_all();
  for (auto& worker : workers_) {
    worker.join();
  }
}

void ThreadPool::execute() {
  std::unique_lock lock(mutex_);
  while (next_chunk_ < chunk_count_) {
    const auto chunk = next_chunk_++;
    const auto count = chunk_count_;
    const auto* task = task_;
    lock.unlock();
    std::exception_ptr error;
    try {
      (*task)(chunk, count);
    } catch (...) {
      error = std::current_exception();
    }
    lock.lock();
    if (error && !error_) {
      error_ = error;
    }
    ++finished_chunks_;
    if (finished_chunks_ == chunk_count_) {
      done_.notify_one();
    }
  }
}

void ThreadPool::work() {
  std::size_t seen = 0;
  std::unique_lock lock(mutex_);
  for (;;) {
    wake_.wait(lock, [&] { return stopping_ || generation_ != seen; });
    if (stopping_) {
      return;
    }
    seen = generation_;
    lock.unlock();
    execute();
    lock.lock();
  }
}

void ThreadPool::run(std::size_t chunk_count,
                     const std::function<void(std::size_t, std::size_t)>& task) {
  if (workers_.empty()) {
    std::exception_ptr error;
    for (std::size_t chunk = 0; chunk < chunk_count; ++chunk) {
      try {
        task(chunk, chunk_count);
      } catch (...) {
        if (!error) {
          error = std::current_exception();
        }
      }
    }
    if (error) {
      std::rethrow_exception(error);
    }
    return;
  }
  {
    const std::scoped_lock lock(mutex_);
    task_ = &task;
    chunk_count_ = chunk_count;
    next_chunk_ = 0;
    finished_chunks_ = 0;
    error_ = nullptr;
    ++generation_;
  }
  wake_.notify_all();
  execute();
  std::unique_lock lock(mutex_);
  done_.wait(lock, [&] { return finished_chunks_ == chunk_count_; });
  task_ = nullptr;
  if (error_) {
    std::rethrow_exception(error_);
  }
}

}  // namespace spindle
