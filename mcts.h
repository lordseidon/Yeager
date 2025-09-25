#pragma once

#include "position.h"
#include <vector>
#include <cmath>
#include <memory>

class MCTSNode {
public:
    MCTSNode(Position& pos, Move move = Move(), MCTSNode* parent = nullptr);

    bool is_fully_expanded() const;
    MCTSNode* best_child(double c_param = 1.414) const;
    MCTSNode* expand(Position& pos);
    double rollout(Position& pos);
    double get_wins() const { return wins; }
    void backpropagate(double result);

public:
    Move move;
    MCTSNode* parent;
    std::vector<std::unique_ptr<MCTSNode>> children;
    int visits;
private:
    double wins;
    std::vector<Move> untried_moves;
};

Move mcts_search(Position& initial_pos, int iterations);
