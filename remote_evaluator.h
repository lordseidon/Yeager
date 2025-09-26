#pragma once

#include <vector>
#include <future>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "inference.grpc.pb.h" // Generated from proto

// Struct to hold the result from the neural network
struct EvaluationResult {
    std::vector<float> policy;
    float value;
};

// Struct for a single request from an MCTS thread
struct InferenceRequest {
    std::vector<float> position_tensor; // Flattened 8*8*18
    std::promise<EvaluationResult> promise;
};

class RemoteEvaluator {
public:
    RemoteEvaluator(const std::string& server_address, int batch_size);
    ~RemoteEvaluator();

    // MCTS threads call this to get an evaluation
    std::future<EvaluationResult> queue_request(std::vector<float>&& position_tensor);

private:
    void processing_loop(); // The dedicated networking thread runs this

    const int batch_size_;
    std::unique_ptr<inference::InferenceService::Stub> stub_;

    std::vector<InferenceRequest> request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    
    std::thread processing_thread_;
    bool stop_thread_;
};