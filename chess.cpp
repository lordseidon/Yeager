#include "position.h"
#include "mcts.h"
#include <iostream>
// #include "position_tensor.h"
#include "fen_to_tensor.h"


int main() {
    initialise_all_databases();
	zobrist::initialise_zobrist_keys();


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

    std::string starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    int game_id = 101;
    
    std::cout << "--- Processing FEN with integer ID: " << game_id << " ---" << std::endl;
    
    // The 'auto' keyword automatically deduces the type of 'result1' to be TensorResult<int>
    auto result1 = fen_to_tensor(starting_fen, game_id);

    // Access the results using the struct members .id and .tensor
    std::cout << "Successfully processed ID: " << result1.id << std::endl;
    std::cout << "Tensor size: " << result1.tensor.size() << std::endl;
    
    // Pass the tensor to a print function for verification
    print_tensor_summary(result1.tensor);
    
    std::cout << "\n\n=================================================\n\n";

    // --- Example 2: Using a string ID with a mid-game position ---
    std::string match_id = "Rapid_Game_vs_PlayerX";
    std::string mid_game_fen = "r1bqk2r/pp2ppbp/2np1np1/8/3NP3/2N1B3/PPPQ1PPP/R3KB1R w KQkq - 0 8";

    std::cout << "--- Processing FEN with string ID: \"" << match_id << "\" ---" << std::endl;

    // Here, 'auto' will deduce the type as TensorResult<std::string>
    auto result2 = fen_to_tensor(mid_game_fen, match_id);
    
    std::cout << "Successfully processed ID: \"" << result2.id << "\"" << std::endl;
    
    // Print the detailed structured view for this more complex position
    print_full_tensor_structured(result2.tensor);


    return 0;
}
