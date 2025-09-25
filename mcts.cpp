#include "mcts.h"
#include "types.h"
#include "evaluator.h"
#include <limits>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>

MCTSNode::MCTSNode(Position& pos, Move move, MCTSNode* parent)
    : move(move), parent(parent), visits(0), wins(0) {
    if (pos.turn() == WHITE) {
        MoveList<WHITE> moves(pos);
        for (const auto& m : moves) {
            untried_moves.push_back(m);
        }
    } else {
        MoveList<BLACK> moves(pos);
        for (const auto& m : moves) {
            untried_moves.push_back(m);
        }
    }
}

bool MCTSNode::is_fully_expanded() const {
    return untried_moves.empty();
}

MCTSNode* MCTSNode::best_child(double c_param) const {
    double best_score = -std::numeric_limits<double>::max();
    MCTSNode* best_child = nullptr;

    for (const auto& child : children) {
        double score = child->wins / child->visits + c_param * std::sqrt(std::log(visits) / child->visits);
        if (score > best_score) {
            best_score = score;
            best_child = child.get();
        }
    }
    return best_child;
}

MCTSNode* MCTSNode::expand(Position& pos) {
    if (pos.turn() == WHITE) {
        MoveList<WHITE> moves(pos);
        int move_index = std::rand() % untried_moves.size();
        Move expanding_move = untried_moves[move_index];
        untried_moves.erase(untried_moves.begin() + move_index);

        pos.play<WHITE>(expanding_move);
        children.push_back(std::make_unique<MCTSNode>(pos, expanding_move, this));
    } else {
        MoveList<BLACK> moves(pos);
        int move_index = std::rand() % untried_moves.size();
        Move expanding_move = untried_moves[move_index];
        untried_moves.erase(untried_moves.begin() + move_index);

        pos.play<BLACK>(expanding_move);
        children.push_back(std::make_unique<MCTSNode>(pos, expanding_move, this));
    }
    return children.back().get();
}

double MCTSNode::rollout(Position& pos) {
    // Check for terminal states first
    if (pos.turn() == WHITE) {
        if (is_checkmate(pos)) return -1.0; // Loss for WHITE
        if (is_stalemate(pos)) return 0.0;  // Draw
        MoveList<WHITE> moves(pos);
        if (moves.size() == 0) return 0.0; // Draw
    } else {
        if (is_checkmate(pos)) return 1.0; // Win for WHITE
        if (is_stalemate(pos)) return 0.0; // Draw
        MoveList<BLACK> moves(pos);
        if (moves.size() == 0) return 0.0; // Draw
    }
    
    // Use evaluator as value network - directly evaluate the position
    int eval = evaluator(pos);
    double normalized_eval = (double)eval / 3000.0;
    // Clamp to [-1, 1] range
    if (normalized_eval > 1.0) normalized_eval = 1.0;
    if (normalized_eval < -1.0) normalized_eval = -1.0;
    
    std::cout << "Value network evaluation: " << eval << ", normalized: " << normalized_eval << std::endl;
    return normalized_eval;
}

void MCTSNode::backpropagate(double result) {
    visits++;
    wins += result;
    if (parent) {
        parent->backpropagate(-result);
    }
}

Move mcts_search(Position& initial_pos, int iterations) {
    std::cout << "Starting MCTS search..." << std::endl;
    MCTSNode root(initial_pos);

    for (int i = 0; i < iterations; ++i) {
        std::cout << "Iteration " << i << std::endl;
        Position pos = initial_pos;
        MCTSNode* node = &root;

        // Selection
        while (node->is_fully_expanded() && !node->children.empty()) {
            node = node->best_child();
            if (pos.turn() == WHITE) {
                pos.play<WHITE>(node->move);
            } else {
                pos.play<BLACK>(node->move);
            }
        }

        // Expansion
        if (!node->is_fully_expanded()) {
            node = node->expand(pos);
        }

        // Simulation
        double result = node->rollout(pos);

        // Backpropagation
        node->backpropagate(result);
    }

    // Show top ranking moves
    std::cout << "\n=== MCTS MOVE RANKINGS ===" << std::endl;
    std::vector<MCTSNode*> child_ptrs;
    for (const auto& child : root.children) {
        child_ptrs.push_back(child.get());
    }
    
    // Sort by win rate (descending)
    std::sort(child_ptrs.begin(), child_ptrs.end(), 
        [](const MCTSNode* a, const MCTSNode* b) {
            double win_rate_a = (a->visits > 0) ? a->get_wins() / a->visits : 0.0;
            double win_rate_b = (b->visits > 0) ? b->get_wins() / b->visits : 0.0;
            return win_rate_a > win_rate_b;
        });
    
    // Display top moves
    for (size_t i = 0; i < child_ptrs.size() && i < 10; ++i) {
        const MCTSNode* child = child_ptrs[i];
        double win_rate = (child->visits > 0) ? child->get_wins() / child->visits : 0.0;
        std::cout << (i+1) << ". Move: " << child->move 
                  << " | Visits: " << child->visits 
                  << " | Win Rate: " << std::fixed << std::setprecision(4) << win_rate
                  << " | Score: " << std::fixed << std::setprecision(2) << child->get_wins()
                  << std::endl;
    }
    std::cout << "=========================" << std::endl;

    return root.best_child(0.0)->move;
}