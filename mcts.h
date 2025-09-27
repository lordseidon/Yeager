#pragma once

#include "position.h" // Your existing position/move classes
#include "types.h"
#include "remote_evaluator.h" // The RemoteEvaluator you created
#include <deque>
#include <memory>
#include <atomic>
#include <mutex>

// Forward declare MCTS class for friend declaration
class MCTS;

// MCTS Configuration struct for hyperparameters
struct MCTSConfig {
    int num_threads = 8;           // Number of search threads to run in parallel
    int batch_size = 64;           // Batch size for the RemoteEvaluator
    double c_puct = 4.0;           // PUCT exploration constant
    double temperature = 1.0;      // Temperature for move selection
    double dirichlet_alpha = 0.3;  // Dirichlet noise alpha
    double dirichlet_epsilon = 0.25; // Dirichlet noise mixing ratio
    bool verbose = true;
};

class MCTSNode {
public:
    // Constructor now takes a policy prior
    MCTSNode(Move move, MCTSNode* parent, float policy_prior);

    // Selects the best child according to the PUCT formula
    MCTSNode* best_child(double c_puct) const;

    // Expands this node, creating children for all legal moves
    // Reverted to Position& because move generation is not const-correct.
    void expand(Position& pos, const std::deque<float>& policy_priors);

    // Recursively updates statistics up the tree
    void backpropagate(double value);

    // Calculates the mean action value (Q) for this node
    double q_value() const;

public:
    Move move;
    MCTSNode* parent;
    std::deque<std::unique_ptr<MCTSNode>> children;

    float policy_prior_;
    std::atomic<int> visits{0};

private:
    // Grant MCTS class access to private members (like expansion_mutex_)
    friend class MCTS; 

    std::atomic<double> total_action_value_{0.0};
    // Mutex to ensure a node is expanded by only one thread
    std::mutex expansion_mutex_; 
};


// The main MCTS engine class that orchestrates the search
class MCTS {
public:
    MCTS(MCTSConfig config, std::unique_ptr<RemoteEvaluator> evaluator);

    // Main entry point to find the best move for a given position
    Move run_search(const Position& initial_pos, int iterations);

private:
    struct PendingEvaluation {
        MCTSNode* node;
        Position pos;
        std::future<EvaluationResult> future;
        std::deque<MCTSNode*> path;
    };
    std::deque<PendingEvaluation> pending_evaluations_;

    // The main loop that each search thread will execute
    void search_worker(const Position& root_pos, MCTSNode* root);

    // Selects the final move based on visit counts and temperature
    Move select_best_move(const MCTSNode& root) const;

    // Adds Dirichlet noise to the root node's children for exploration
    void add_dirichlet_noise(MCTSNode& root);
    
private:
    MCTSConfig config_;
    std::unique_ptr<RemoteEvaluator> evaluator_;
};