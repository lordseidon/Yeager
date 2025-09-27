// debug_grpc.cpp - Simple test to isolate the connection issue
#include <grpcpp/grpcpp.h>
#include <iostream>
#include <chrono>
#include "inference.grpc.pb.h"

int main() {
    std::cout << "=== DEBUGGING GRPC CONNECTION ===" << std::endl;
    
    // Test connection
    std::string server_address = "localhost:50055";  // Make sure this matches your server
    std::cout << "Attempting to connect to: " << server_address << std::endl;
    
    auto channel = grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
    auto stub = inference::InferenceService::NewStub(channel);
    
    std::cout << "Channel created, testing connectivity..." << std::endl;
    
    // Test channel state
    auto state = channel->GetState(true);
    std::cout << "Channel state: " << state << std::endl;
    
    // Wait for connection
    std::cout << "Waiting for channel to be ready..." << std::endl;
    auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(10);
    bool connected = channel->WaitForConnected(deadline);
    
    if (!connected) {
        std::cout << "ERROR: Could not connect to server within 10 seconds!" << std::endl;
        std::cout << "Make sure your server is running on " << server_address << std::endl;
        return 1;
    }
    
    std::cout << "✓ Channel connected successfully!" << std::endl;
    
    // Create a simple test request
    inference::BatchInferenceRequest request;
    auto* position = request.add_positions();
    
    // Add dummy data (8*8*18 = 1152 floats)
    for (int i = 0; i < 1152; i++) {
        position->mutable_board_state()->Add(0.5f);
    }
    
    std::cout << "Created test request with " << request.positions_size() << " positions" << std::endl;
    std::cout << "Position 0 has " << request.positions(0).board_state_size() << " values" << std::endl;
    
    // Make the call
    inference::BatchInferenceResponse response;
    grpc::ClientContext context;
    
    // Set a timeout
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(30));
    
    std::cout << "Making gRPC call..." << std::endl;
    grpc::Status status = stub->Evaluate(&context, request, &response);
    
    if (status.ok()) {
        std::cout << "✓ SUCCESS: gRPC call completed!" << std::endl;
        std::cout << "Response has " << response.evaluations_size() << " evaluations" << std::endl;
        
        if (response.evaluations_size() > 0) {
            const auto& eval = response.evaluations(0);
            std::cout << "First evaluation: value=" << eval.value() 
                      << ", policy_size=" << eval.policy_logits_size() << std::endl;
        }
    } else {
        std::cout << "✗ ERROR: gRPC call failed!" << std::endl;
        std::cout << "Error code: " << status.error_code() << std::endl;
        std::cout << "Error message: " << status.error_message() << std::endl;
        std::cout << "Error details: " << status.error_details() << std::endl;
        
        // Print specific error types
        switch (status.error_code()) {
            case grpc::StatusCode::UNIMPLEMENTED:
                std::cout << "→ Method not implemented on server" << std::endl;
                break;
            case grpc::StatusCode::NOT_FOUND:
                std::cout << "→ Method or service not found" << std::endl;
                break;
            case grpc::StatusCode::UNAVAILABLE:
                std::cout << "→ Server unavailable" << std::endl;
                break;
            case grpc::StatusCode::DEADLINE_EXCEEDED:
                std::cout << "→ Request timed out" << std::endl;
                break;
            default:
                std::cout << "→ Other error" << std::endl;
        }
        
        return 1;
    }
    
    return 0;
}