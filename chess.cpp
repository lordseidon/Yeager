#include "position.h"
#include "mcts.h"
#include <iostream>

int main() {
    // Initialize a starting chess position (you may need to adjust this
    // depending on how your Position class is implemented).
    Position pos;  
    Position::set("1q5k/1b3Qp1/3bN1p1/2p5/1nP1p3/1P2P3/6PP/B5K1 w - - 0 28", pos);

    int iterations = 15000;  // number of MCTS iterations to run

    std::cout << "Running MCTS with " << iterations << " iterations..." << std::endl;

    Move bestMove = mcts_search(pos, iterations);

    std::cout << "Best move found: " << bestMove << std::endl;

    return 0;
}
