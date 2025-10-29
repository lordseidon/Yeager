#pragma once
#include <deque>
#include <future>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <string>
#include <vector>

struct _object;
typedef _object PyObject;
struct _ts;
typedef _ts PyThreadState;

struct EvaluationResult {
    std::vector<float> policy;
    float value;
};

struct InferenceRequest {
    std::vector<float> position_tensor;
    std::promise<EvaluationResult> promise;
    
    InferenceRequest(const InferenceRequest&) = delete;
    InferenceRequest& operator=(const InferenceRequest&) = delete;
    InferenceRequest(InferenceRequest&&) = default;
    InferenceRequest& operator=(InferenceRequest&&) = default;
    InferenceRequest() = default;
};

class RemoteEvaluator {
public:
    RemoteEvaluator(const std::string& model_path, int batch_size, int min_batch_size = 64);
    ~RemoteEvaluator();

    std::future<EvaluationResult> queue_request(std::vector<float>&& position_tensor);

private:
    void processing_loop();
    void initialize_python_model(const std::string& model_path);
    static void PyAssert(bool result);

    const int batch_size_;
    const int min_batch_size_;
    
    PyObject* python_evaluator_;  // ChessEvaluator instance
    PyObject* device_worker_;     // TPUDeviceWorker instance

    std::deque<InferenceRequest> request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    
    std::thread processing_thread_;
    std::atomic<bool> stop_thread_;
    
    PyThreadState* main_thread_state_;
    
    std::chrono::steady_clock::time_point last_batch_time_;
    static constexpr int BATCH_TIMEOUT_MS = 1;
};