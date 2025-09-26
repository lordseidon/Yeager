#include "position.h"
#include "mcts.h"
#include <iostream>

int main() {
    initialise_all_databases();
	zobrist::initialise_zobrist_keys();


    Position pos;  
    Position::set("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", pos);

    MoveList<WHITE> moves(pos);
    std::cout << "Available moves: " << moves.size() << std::endl;

    int iterations = 60000;  // number of MCTS iterations to run

    // std::cout << "Running MCTS with " << iterations << " iterations..." << std::endl;

    Move bestMove = mcts_search(pos, iterations);

    std::cout << "Best move found: " << bestMove << std::endl;

    return 0;
}
