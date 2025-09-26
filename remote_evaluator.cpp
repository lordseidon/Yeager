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

std::future<EvaluationResult> RemoteEvaluator::queue_request(std::vector<float>&& position_tensor) {
    InferenceRequest req;
    req.position_tensor = std::move(position_tensor);
    std::future<EvaluationResult> future = req.promise.get_future();

    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        request_queue_.emplace_back(std::move(req));
    }
    cv_.notify_one(); // Notify the processing thread that a new item is available

    return future;
}

void RemoteEvaluator::processing_loop() {
    while (true) {
        std::vector<InferenceRequest> batch;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            // Wait until the queue has a full batch or we are stopping
            cv_.wait(lock, [this] {
                return request_queue_.size() >= batch_size_ || stop_thread_;
            });

            if (stop_thread_ && request_queue_.empty()) {
                return;
            }

            // Move requests from the main queue to our local batch
            auto num_to_move = std::min((size_t)batch_size_, request_queue_.size());
            batch.insert(batch.end(),
                         std::make_move_iterator(request_queue_.begin()),
                         std::make_move_iterator(request_queue_.begin() + num_to_move));
            request_queue_.erase(request_queue_.begin(), request_queue_.begin() + num_to_move);
        }

        if (batch.empty()) continue;

        // Prepare the gRPC request
        inference::BatchInferenceRequest grpc_request;
        for (const auto& req : batch) {
            auto* pos_tensor = grpc_request.add_positions();
            pos_tensor->mutable_board_state()->Add(req.position_tensor.begin(), req.position_tensor.end());
        }

        // Send request and get response
        inference::BatchInferenceResponse grpc_response;
        grpc::ClientContext context;
        grpc::Status status = stub_->Evaluate(&context, grpc_request, &grpc_response);

        // Fulfill the promises to unblock the waiting MCTS threads
        if (status.ok()) {
            for (size_t i = 0; i < batch.size(); ++i) {
                const auto& eval = grpc_response.evaluations(i);
                EvaluationResult result;
                result.value = eval.value();
                result.policy.assign(eval.policy_logits().begin(), eval.policy_logits().end());
                batch[i].promise.set_value(std::move(result));
            }
        } else {
            // Handle error: e.g., fulfill promises with an exception
            for (size_t i = 0; i < batch.size(); ++i) {
                try {
                    throw std::runtime_error("gRPC call failed: " + status.error_message());
                } catch (...) {
                    batch[i].promise.set_exception(std::current_exception());
                }
            }
        }
    }
}