// grpc_reflection_test.cpp - Check what the server actually exposes
#include <grpcpp/grpcpp.h>
#include <iostream>
#include "inference.grpc.pb.h"

void print_service_info() {
    std::cout << "=== CLIENT EXPECTS ===" << std::endl;
    std::cout << "Service: inference.InferenceService" << std::endl;
    std::cout << "Method: Evaluate" << std::endl;
    std::cout << "Package: " << std::endl;
    
    // Print what the client thinks the service should be
    auto descriptor = inference::InferenceService::service_full_name();
    std::cout << "Full service name from client: " << descriptor << std::endl;
}

int main() {
    print_service_info();
    
    std::cout << "\n=== TESTING DIFFERENT METHOD CALLS ===" << std::endl;
    
    std::string server_address = "localhost:50055";
    auto channel = grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
    auto stub = inference::InferenceService::NewStub(channel);
    
    // Wait for connection
    auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(5);
    if (!channel->WaitForConnected(deadline)) {
        std::cout << "Cannot connect to server" << std::endl;
        return 1;
    }
    
    // Create test request
    inference::BatchInferenceRequest request;
    auto* position = request.add_positions();
    for (int i = 0; i < 1152; i++) {
        position->mutable_board_state()->Add(0.5f);
    }
    
    // Test 1: Try the Evaluate method
    std::cout << "\n1. Testing Evaluate method..." << std::endl;
    {
        inference::BatchInferenceResponse response;
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
        
        grpc::Status status = stub->Evaluate(&context, request, &response);
        
        std::cout << "Status: " << (status.ok() ? "SUCCESS" : "FAILED") << std::endl;
        if (!status.ok()) {
            std::cout << "Error code: " << status.error_code() << std::endl;
            std::cout << "Error message: " << status.error_message() << std::endl;
        }
    }
    
    // Test 2: Try raw gRPC call with different service names
    std::cout << "\n2. Testing with raw gRPC call..." << std::endl;
    {
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
        
        // Try to make a raw call to see what happens
        inference::BatchInferenceResponse response;
        
        // This will show us exactly what's being called
        auto call = stub->Evaluate(&context, request, &response);
        
        std::cout << "Raw call status: " << (call.ok() ? "SUCCESS" : "FAILED") << std::endl;
        if (!call.ok()) {
            std::cout << "Raw call error: " << call.error_code() << " - " << call.error_message() << std::endl;
        }
    }
    
    return 0;
}