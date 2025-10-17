#include "position.h"
#include "mcts.h"
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include "remote_evaluator.h"
#include "fen_to_tensor.h"
#include "position_tensor.h"

int main() {
    initialise_all_databases();
    zobrist::initialise_zobrist_keys();
    MoveMappings::initialize_move_mappings();
    
    MCTSConfig config;
    config.num_threads = 24;
    config.batch_size = 256;
    config.c_puct = 3;
    config.temperature = 1.5;
    config.verbose = true;
    config.dirichlet_alpha = 0.05;
    config.dirichlet_epsilon = 0.1;
    
    // Initialize the Remote Evaluator
    std::string server_address = "localhost:50055";
    auto evaluator = std::make_unique<RemoteEvaluator>(server_address, config.batch_size);
    std::cout << "Connected to evaluation server at " << server_address << std::endl;
    
    // Create the MCTS engine instance
    MCTS mcts_engine(config, std::move(evaluator));
    
    // Get user input for FEN string and iterations
    std::string fen_string;
    int num_iterations;
    
    std::cout << "\n=== MCTS Search Engine ===" << std::endl;
    std::cout << "Enter FEN string: ";
    std::getline(std::cin, fen_string);
    std::cout << "Enter number of iterations: ";
    std::cin >> num_iterations;
    
    // Validate iterations input
    if (num_iterations <= 0) {
        std::cerr << "Error: Number of iterations must be positive." << std::endl;
        return 1;
    }
    
    // Set up the position from user input
    Position pos;
    try {
        Position::set(fen_string, pos);
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid FEN string - " << e.what() << std::endl;
        return 1;
    }
    
    Position initial_pos = pos;
    std::cout << "\nStarting search from position: " << initial_pos.fen() << std::endl;
    std::cout << "Running " << num_iterations << " iterations...\n" << std::endl;
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Run the search
    Move best_move = mcts_engine.run_search(initial_pos, num_iterations);
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Calculate statistics
    double seconds = duration.count() / 1000.0;
    double iterations_per_second = num_iterations / seconds;
    
    // Output the results
    std::cout << "\n=== Search Results ===" << std::endl;
    std::cout << "Best move found: " << best_move << std::endl;
    std::cout << "\n=== Performance Statistics ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total time: " << seconds << " seconds (" << duration.count() << " ms)" << std::endl;
    std::cout << "Iterations per second: " << iterations_per_second << std::endl;
    std::cout << "Average time per iteration: " << (duration.count() / static_cast<double>(num_iterations)) << " ms" << std::endl;
    
    // Return 0 for successful execution
    // The best move is output to stdout and will be parsed by game.cpp
    return 0;
}