#pragma once
#include "position.h"
#include "types.h"
#include "remote_evaluator.h"
#include <deque>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>

class MCTS;

struct MCTSConfig {
    int num_threads = 128;
    int batch_size = 128;
    double c_puct = 4.0;
    double temperature = 1.0;
    double dirichlet_alpha = 0.3;
    double dirichlet_epsilon = 0.25;
    bool verbose = true;
};

class MCTSNode {
public:
    MCTSNode(Move move, MCTSNode* parent, float policy_prior);
    MCTSNode* best_child(double c_puct) const;
    void expand(Position& pos, const std::vector<float>& policy_priors);  // Changed from deque to vector
    void backpropagate(double value);
    double q_value() const;

public:
    Move move;
    MCTSNode* parent;
    std::deque<std::unique_ptr<MCTSNode>> children;
    float policy_prior_;
    std::atomic<int> visits{0};
    std::atomic<int> virtual_loss_{0};  // Track virtual losses for parallel MCTS
    std::deque<std::string> fen_history; // Track the 7 previous positions leading to this node

private:
    friend class MCTS;
    std::atomic<double> total_action_value_{0.0};
    std::mutex expansion_mutex_;
};

class MCTS {
public:
    MCTS(MCTSConfig config, std::unique_ptr<RemoteEvaluator> evaluator);
    
    // Main entry point - now with clear_after_search parameter
    Move run_search(const Position& initial_pos, int iterations, bool clear_after_search = true);
    
    // Overload with position history
    Move run_search(const Position& initial_pos, int iterations, bool clear_after_search, const std::deque<std::string>& position_history);
    
    // Explicitly clear evaluator state
    void clear_evaluator_state();
    
    // Dynamic configuration methods
    void set_dirichlet_alpha(double alpha);

private:
    struct PendingEvaluation {
        MCTSNode* node;
        Position pos;
        std::future<EvaluationResult> future;
        std::deque<MCTSNode*> path;
    };
    std::deque<PendingEvaluation> pending_evaluations_;
    
    void search_worker(const Position& root_pos, MCTSNode* root);
    Move select_best_move(const MCTSNode& root) const;
    void add_dirichlet_noise(MCTSNode& root);
    std::vector<double> apply_temperature(const std::vector<double>& visit_counts, double temperature) const;
    MCTSNode* sample_from_distribution(const std::vector<MCTSNode*>& nodes, 
                                       const std::vector<double>& probabilities) const;
    
private:
    MCTSConfig config_;
    std::unique_ptr<RemoteEvaluator> evaluator_;
};