#include "position.h"
#include "mcts.h"
#include <iostream>
// #include "position_tensor.h"
#include "remote_evaluator.h"
#include "fen_to_tensor.h"
#include "position_tensor.h"


int main() {
    initialise_all_databases();
	zobrist::initialise_zobrist_keys();
    MoveMappings::initialize_move_mappings();

    // Position pos;  
    // Position::set("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", pos);

    // MoveList<WHITE> moves(pos);
    // std::cout << "Available moves: " << moves.size() << std::endl;

    // int iterations = 60000;  // number of MCTS iterations to run

    // // std::cout << "Running MCTS with " << iterations << " iterations..." << std::endl;

    // Move bestMove = mcts_search(pos, iterations);

    // std::cout << "Best move found: " << bestMove << std::endl;


// =================================================================================
// =================================================================================
// =================================================================================
// =================================================================================

    // Position pos;
    // Position::set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", pos);
    
    // // Convert position to neural network tensor
    // std::vector<float> tensor = position_to_tensor(pos);
    
    // // Debug: print summary
    // print_tensor_summary(tensor);

    // std::string starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    // int game_id = 101;
    
    // std::cout << "--- Processing FEN with integer ID: " << game_id << " ---" << std::endl;
    
    // // The 'auto' keyword automatically deduces the type of 'result1' to be TensorResult<int>
    // auto result1 = fen_to_tensor(starting_fen, game_id);

    // // Access the results using the struct members .id and .tensor
    // std::cout << "Successfully processed ID: " << result1.id << std::endl;
    // std::cout << "Tensor size: " << result1.tensor.size() << std::endl;
    
    // // Pass the tensor to a print function for verification
    // print_tensor_summary(result1.tensor);
    
    // std::cout << "\n\n=================================================\n\n";

    // // --- Example 2: Using a string ID with a mid-game position ---
    // std::string match_id = "Rapid_Game_vs_PlayerX";
    // std::string mid_game_fen = "r1bqk2r/pp2ppbp/2np1np1/8/3NP3/2N1B3/PPPQ1PPP/R3KB1R w KQkq - 0 8";

    // std::cout << "--- Processing FEN with string ID: \"" << match_id << "\" ---" << std::endl;

    // // Here, 'auto' will deduce the type as TensorResult<std::string>
    // auto result2 = fen_to_tensor(mid_game_fen, match_id);
    
    // std::cout << "Successfully processed ID: \"" << result2.id << "\"" << std::endl;
    
    // // Print the detailed structured view for this more complex position
    // print_full_tensor_structured(result2.tensor);

    MCTSConfig config;
    config.num_threads = 64;
    config.batch_size = 256; // Should match your server's preferred batch size
    config.c_puct = 8;
    config.temperature = 0;
    config.verbose = true;
    config.dirichlet_alpha = 0.3;
    config.dirichlet_epsilon = 0.25;

    // 2. Initialize the Remot e Evaluator
    // This connects to your Python gRPC server.
    // Make sure the Python server is running before you start the C++ client.
    std::string server_address = "localhost:50055";
    auto evaluator = std::make_unique<RemoteEvaluator>(server_address, config.batch_size);
    std::cout << "Connected to evaluation server at " << server_address << std::endl;

    // 3. Create the MCTS engine instance
    MCTS mcts_engine(config, std::move(evaluator));

    // 4. Set up the starting position
     Position pos;
     Position::set("1r3rk1/p1qb1pb1/2p1p1pp/8/2pP4/1P3BP1/PB2QPKR/7R w - - 0 25", pos);
     Position initial_pos = pos; // Copy for safety
    std::cout << "Starting search from position: " << initial_pos.fen() << std::endl;

    // 5. Run the search for a set number of iterations
    int total_iterations = 4500;
    Move best_move = mcts_engine.run_search(initial_pos, total_iterations);

    // 6. Output the result
    std::cout << "\nSearch complete." << std::endl;
    std::cout << "Best move found: " << best_move << std::endl;


    return 0;
}
