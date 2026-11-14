#pragma once
// Bounded background QP-fail dump writer. Owns a single worker thread; callers
// enqueue payloads and join on stop. Never detach; drops are counted.

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

namespace kmpc {

template <class Payload>
class QpFailDumpWorker {
 public:
  using Writer = std::function<void(Payload&&)>;

  explicit QpFailDumpWorker(std::size_t capacity = 4) : capacity_(capacity) {}

  ~QpFailDumpWorker() { stop_join(); }

  QpFailDumpWorker(const QpFailDumpWorker&) = delete;
  QpFailDumpWorker& operator=(const QpFailDumpWorker&) = delete;

  void start(Writer writer) {
    std::lock_guard<std::mutex> lock(mu_);
    if (running_) {
      return;
    }
    writer_ = std::move(writer);
    stop_ = false;
    running_ = true;
    worker_ = std::thread([this] { loop(); });
  }

  // Returns false when the bounded queue is full (payload dropped).
  bool enqueue(Payload payload) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!running_ || stop_) {
      ++drops_;
      return false;
    }
    if (queue_.size() >= capacity_) {
      ++drops_;
      return false;
    }
    queue_.push_back(std::move(payload));
    cv_.notify_one();
    return true;
  }

  void stop_join() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (!running_) {
        return;
      }
      stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
    std::lock_guard<std::mutex> lock(mu_);
    running_ = false;
    queue_.clear();
  }

  uint64_t drops() const {
    std::lock_guard<std::mutex> lock(mu_);
    return drops_;
  }

  bool running() const {
    std::lock_guard<std::mutex> lock(mu_);
    return running_;
  }

  std::size_t queued() const {
    std::lock_guard<std::mutex> lock(mu_);
    return queue_.size();
  }

 private:
  void loop() {
    for (;;) {
      Payload item;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
        if (stop_ && queue_.empty()) {
          return;
        }
        item = std::move(queue_.front());
        queue_.pop_front();
      }
      if (writer_) {
        writer_(std::move(item));
      }
    }
  }

  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Payload> queue_;
  Writer writer_;
  std::thread worker_;
  std::size_t capacity_ = 4;
  bool stop_ = false;
  bool running_ = false;
  uint64_t drops_ = 0;
};

}  // namespace kmpc
