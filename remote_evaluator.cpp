#include "remote_evaluator.h"
#include <grpcpp/grpcpp.h>

// g++ chess.cpp remote_evaluator.cpp fen_to_tensor.cpp position.cpp types.cpp tables.cpp -o chess.exe -mconsole && chess.exe

RemoteEvaluator::RemoteEvaluator(const std::string& server_address, int batch_size)
    : batch_size_(batch_size), stop_thread_(false) {
    
    auto channel = grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
    stub_ = inference::InferenceService::NewStub(channel);

    // Start the dedicated thread for processing batches
    processing_thread_ = std::thread(&RemoteEvaluator::processing_loop, this);
}

RemoteEvaluator::~RemoteEvaluator() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_thread_ = true;
    }
    cv_.notify_one();
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
}

std::future<EvaluationResult> RemoteEvaluator::queue_request(std::deque<float>&& position_tensor) {
    InferenceRequest req;
    req.position_tensor = std::move(position_tensor);
    std::future<EvaluationResult> future = req.promise.get_future();

    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        request_queue_.emplace_back(std::move(req));
        // std::cout << "[CLIENT] Queued request, queue size now: " << request_queue_.size() << std::endl;
    }
    cv_.notify_one();

    return future;
}

// In remote_evaluator.cpp, replace your existing processing_loop with:
void RemoteEvaluator::processing_loop() {
    const int MAX_BATCH = 200;  // Keep batches at this size
    const int MIN_BATCH = 128;   // Minimum before sending
    
    while (true) {
        std::deque<InferenceRequest> batch;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            // Wait for minimum batch or timeout
            auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
            bool timed_out = !cv_.wait_until(lock, deadline, [this] {
                return request_queue_.size() >= MIN_BATCH || stop_thread_;
            });
            
            if (stop_thread_ && request_queue_.empty()) {
                return;
            }
            
            // Send batch if we have minimum size OR timeout occurred with any requests
            if (request_queue_.size() >= MIN_BATCH || (timed_out && !request_queue_.empty())) {
                // CRITICAL: Cap at MAX_BATCH to avoid server issues
                size_t num_to_move = std::min((size_t)MAX_BATCH, request_queue_.size());
                
                for (size_t i = 0; i < num_to_move; ++i) {
                    batch.emplace_back(std::move(request_queue_[i]));
                }
                request_queue_.erase(request_queue_.begin(), request_queue_.begin() + num_to_move);
                
                std::cout << "[CLIENT] Sending batch of size " << batch.size() 
                          << " (queue remaining: " << request_queue_.size() << ")" << std::endl;
            }
        }
        
        if (batch.empty()) continue;

        // Prepare gRPC request
        inference::BatchInferenceRequest grpc_request;
        for (const auto& req : batch) {
            auto* pos_tensor = grpc_request.add_positions();
            pos_tensor->mutable_board_state()->Add(req.position_tensor.begin(), req.position_tensor.end());
        }

        // Send request
        inference::BatchInferenceResponse grpc_response;
        grpc::ClientContext context;
        
        // Set timeout for the RPC call
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(30));
        
        grpc::Status status = stub_->Evaluate(&context, grpc_request, &grpc_response);

        if (!status.ok()) {
            std::cerr << "[ERROR] gRPC failed: " << status.error_message() << std::endl;
            for (auto& req : batch) {
                req.promise.set_exception(std::make_exception_ptr(
                    std::runtime_error("gRPC failed: " + status.error_message())));
            }
            continue;
        }

        if (grpc_response.evaluations_size() != static_cast<int>(batch.size())) {
            std::cerr << "[ERROR] Response mismatch: expected " << batch.size() 
                      << ", got " << grpc_response.evaluations_size() << std::endl;
            for (auto& req : batch) {
                req.promise.set_exception(std::make_exception_ptr(
                    std::runtime_error("Response size mismatch")));
            }
            continue;
        }

        // Process responses
        for (size_t i = 0; i < batch.size(); ++i) {
            const auto& eval = grpc_response.evaluations(i);
            EvaluationResult result;
            result.value = eval.value();
            result.policy.assign(eval.policy_logits().begin(), eval.policy_logits().end());
            batch[i].promise.set_value(std::move(result));
        }
    }
}
