#include "position.h"
#include "mcts.h"
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <Python.h>
#include "remote_evaluator.h"
#include "fen_to_tensor.h"
#include "position_tensor.h"

int main() {
    std::cout << "=== Chess Engine Starting ===" << std::endl;
    
    // Initialize Python interpreter before anything else
    std::cout << "Initializing Python interpreter..." << std::endl;
    if (!Py_IsInitialized()) {
        Py_Initialize();
        PyEval_InitThreads();
    }
    std::cout << "Python initialized" << std::endl;
    
    // Initialize chess databases
    std::cout << "Initializing chess databases..." << std::endl;
    initialise_all_databases();
    zobrist::initialise_zobrist_keys();
    MoveMappings::initialize_move_mappings();
    std::cout << "Chess databases initialized" << std::endl;
    
    // MCTS configuration
    MCTSConfig config;
    config.num_threads = 96;
    config.batch_size = 256;
    config.c_puct = 3.0;
    config.temperature = 1.5;
    config.verbose = true;
    config.dirichlet_alpha = 0.25;
    config.dirichlet_epsilon = 0.2;
    
    std::string model_path = "model";  // Will look for model.py in current dir or ./server
    int min_batch_size = 16;
    
    std::cout << "\n=== Configuration ===" << std::endl;
    std::cout << "Model path: " << model_path << std::endl;
    std::cout << "Threads: " << config.num_threads << std::endl;
    std::cout << "Max batch size: " << config.batch_size << std::endl;
    std::cout << "Min batch size: " << min_batch_size << std::endl;
    std::cout << "C-PUCT: " << config.c_puct << std::endl;
    std::cout << "Temperature: " << config.temperature << std::endl;
    
    try {
        // Create evaluator (this loads the TensorFlow model)
        std::cout << "\nInitializing remote evaluator..." << std::endl;
        auto evaluator = std::make_unique<RemoteEvaluator>(model_path, config.batch_size, min_batch_size);
        std::cout << "Evaluator initialized successfully\n" << std::endl;
        
        // Create MCTS engine
        std::cout << "Creating MCTS engine..." << std::endl;
        MCTS mcts_engine(config, std::move(evaluator));
        std::cout << "MCTS engine created\n" << std::endl;
        
        // Set up test position
        std::string fen_string = "rn3r2/pp2qpk1/2p3p1/5P2/2BP2Q1/2b1P3/P4P2/R1B1R1K1 w - - 2 19";
        int num_iterations = 800;
        
        Position pos;
        Position::set(fen_string, pos);
        
        std::cout << "=== Search Configuration ===" << std::endl;
        std::cout << "Position FEN: " << pos.fen() << std::endl;
        std::cout << "Iterations: " << num_iterations << std::endl;
        std::cout << "Starting search...\n" << std::endl;
        
        // Run search with timing
        Move best_move = mcts_engine.run_search(pos, num_iterations);
        auto start_time = std::chrono::high_resolution_clock::now();
        best_move = mcts_engine.run_search(pos, num_iterations);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Display results
        std::cout << "\n=== SEARCH RESULTS ===" << std::endl;
        std::cout << "Best move: " << best_move << std::endl;
        std::cout << "Search time: " << duration.count() << " ms" << std::endl;
        std::cout << "Search speed: " << std::fixed << std::setprecision(2) 
                  << (num_iterations * 1000.0 / duration.count()) << " iterations/second" << std::endl;
        
        if (duration.count() > 0) {
            std::cout << "Time per iteration: " << std::fixed << std::setprecision(2)
                      << (duration.count() / static_cast<double>(num_iterations)) << " ms" << std::endl;
        }
        
        // Allow time for all async operations to complete
        std::cout << "\nWaiting for cleanup..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        std::cout << "\n=== Cleanup Starting ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n!!! ERROR !!!" << std::endl;
        std::cerr << "Exception: " << e.what() << std::endl;
        
        // Don't finalize Python on error - let it clean up naturally
        return 1;
    }
    
    std::cout << "=== Chess Engine Shutdown Complete ===" << std::endl;
    
    // Note: We intentionally don't call Py_FinalizeEx() here
    // The RemoteEvaluator destructor handles Python cleanup
    // Calling Py_FinalizeEx() can cause issues with TensorFlow/CUDA
    
    return 0;
}