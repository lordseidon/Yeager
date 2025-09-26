#include "mcts.h"
#include <cmath>
#include <limits>
#include <random>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <thread>

// =================================================================
// PLACEHOLDER FUNCTIONS - YOU MUST IMPLEMENT THESE
// =================================================================

// Maps a move object to an integer index for the policy vector (0 to 4479)
// This is a CRITICAL function you must design based on your move encoding.
int get_policy_index_for_move(const Move& move) {
    // Example: return (from_square * 64) + to_square;
    // The actual implementation depends entirely on your model's output format.
    // Return -1 for an invalid or unmappable move.
    return 0; // Placeholder
}

// Converts your board `Position` object into a flattened 1D vector of floats
// with the shape (8*8*18 = 1152) that your neural network expects.
std::vector<float> convert_position_to_tensor(const Position& pos) {
    // Your implementation here...
    return std::vector<float>(8 * 8 * 18, 0.0f); // Placeholder
}

// =================================================================
// MCTSNode IMPLEMENTATION
// =================================================================

MCTSNode::MCTSNode(Move move, MCTSNode* parent, float policy_prior)
    : move(move), parent(parent), policy_prior_(policy_prior) {}

double MCTSNode::q_value() const {
    int v = visits.load(std::memory_order_relaxed);
    if (v == 0) {
        return 0.0;
    }
    return total_action_value_.load(std::memory_order_relaxed) / v;
}

MCTSNode* MCTSNode::best_child(double c_puct) const {
    MCTSNode* best = nullptr;
    double max_score = -std::numeric_limits<double>::max();

    int parent_visits = visits.load(std::memory_order_relaxed);
    
    for (const auto& child : children) {
        // PUCT formula: Q(s,a) + U(s,a)
        double q = child->q_value();
        double u = c_puct * child->policy_prior_ * (std::sqrt(parent_visits) / (1 + child->visits.load(std::memory_order_relaxed)));
        
        double score = q + u;
        if (score > max_score) {
            max_score = score;
            best = child.get();
        }
    }
    return best;
}

void MCTSNode::expand(Position& pos, const std::vector<float>& policy_priors) {
    // This function assumes it's called within a lock.
    if (pos.turn() == WHITE) {
        MoveList<WHITE> legal_moves(pos);
        for (const auto& m : legal_moves) {
            int policy_index = get_policy_index_for_move(m);
            float prior = (policy_index != -1) ? policy_priors[policy_index] : 0.0f;
            children.push_back(std::make_unique<MCTSNode>(m, this, prior));
        }
    } else {
        MoveList<BLACK> legal_moves(pos);
        for (const auto& m : legal_moves) {
            int policy_index = get_policy_index_for_move(m);
            float prior = (policy_index != -1) ? policy_priors[policy_index] : 0.0f;
            children.push_back(std::make_unique<MCTSNode>(m, this, prior));
        }
    }
}

void MCTSNode::backpropagate(double value) {
    MCTSNode* node = this;
    while (node != nullptr) {
        node->visits.fetch_add(1, std::memory_order_relaxed);
        node->total_action_value_.fetch_add(value, std::memory_order_relaxed);
        value = -value; // The value is from the perspective of the other player
        node = node->parent;
    }
}

// =================================================================
// MCTS IMPLEMENTATION
// =================================================================

MCTS::MCTS(MCTSConfig config, std::unique_ptr<RemoteEvaluator> evaluator)
    : config_(std::move(config)), evaluator_(std::move(evaluator)) {}

Move MCTS::run_search(const Position& initial_pos, int iterations) {
    MCTSNode root(Move(), nullptr, 1.0f);

    // Initial evaluation of the root node is required to expand it
    std::vector<float> root_tensor = convert_position_to_tensor(initial_pos);
    EvaluationResult root_eval = evaluator_->queue_request(std::move(root_tensor)).get();
    
    { // Lock is not strictly needed for root, but good practice
        std::scoped_lock lock(root.expansion_mutex_);
        root.expand(Position(initial_pos), root_eval.policy);
    }
    
    // The value is from white's perspective, adjust if black to move.
    double root_value = (initial_pos.turn() == WHITE) ? root_eval.value : -root_eval.value;
    root.backpropagate(root_value);

    add_dirichlet_noise(root);

    // Launch worker threads
    std::vector<std::thread> threads;
    threads.reserve(config_.num_threads);
    int iterations_per_thread = iterations / config_.num_threads;

    for (int i = 0; i < config_.num_threads; ++i) {
        threads.emplace_back([this, iterations_per_thread, &initial_pos, &root]() {
            for (int j = 0; j < iterations_per_thread; ++j) {
                this->search_worker(initial_pos, &root);
            }
        });
    }

    // Wait for all threads to finish
    for (auto& t : threads) {
        t.join();
    }

    return select_best_move(root);
}

void MCTS::search_worker(const Position& root_pos, MCTSNode* root) {
    Position pos = root_pos;
    MCTSNode* node = root;

    // 1. SELECTION: Traverse the tree using PUCT
    while (!node->children.empty()) {
        node = node->best_child(config_.c_puct);
        if (pos.turn() == WHITE) pos.play<WHITE>(node->move);
        else pos.play<BLACK>(node->move);
    }

    // Check for a terminal game state
    if (is_checkmate(pos) || is_stalemate(pos)) {
        double result = (is_checkmate(pos)) ? -1.0 : 0.0; // Loss for current player
        node->backpropagate(result);
        return;
    }

    // 2. EXPANSION & EVALUATION
    // Acquire a lock to ensure this node is only expanded once.
    std::scoped_lock lock(node->expansion_mutex_);
    
    // It's possible another thread expanded this node while we were waiting for the lock.
    // If children is not empty, the node has already been processed.
    if (node->children.empty()) {
        std::vector<float> pos_tensor = convert_position_to_tensor(pos);
        std::future<EvaluationResult> future_eval = evaluator_->queue_request(std::move(pos_tensor));
        EvaluationResult eval = future_eval.get();

        node->expand(pos, eval.policy);
        
        double value = eval.value; // Value is from the current player's perspective.
        node->backpropagate(value);
    }
}

Move MCTS::select_best_move(const MCTSNode& root) const {
    if (root.children.empty()) return Move(); // Should not happen

    MCTSNode* best_move_node = nullptr;
    int max_visits = -1;

    // In tournament play or late in the game, you'd pick the move with the most visits (temp=0).
    // For training, you can sample from a distribution based on visit counts.
    for (const auto& child : root.children) {
        if (child->visits > max_visits) {
            max_visits = child->visits;
            best_move_node = child.get();
        }
        if (config_.verbose) {
            std::cout << "Move: " << child->move 
                      << " | Visits: " << child->visits 
                      << " | Win Rate: " << std::fixed << std::setprecision(4) << child->q_value()
                      << std::endl;
        }
    }
    return best_move_node->move;
}

void MCTS::add_dirichlet_noise(MCTSNode& root) {
    if (config_.dirichlet_epsilon == 0.0 || root.children.empty()) return;

    std::vector<double> noise;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::gamma_distribution<double> gamma(config_.dirichlet_alpha, 1.0);
    
    double noise_sum = 0.0;
    for (size_t i = 0; i < root.children.size(); ++i) {
        double n = gamma(gen);
        noise.push_back(n);
        noise_sum += n;
    }
    
    for (size_t i = 0; i < root.children.size(); ++i) {
        root.children[i]->policy_prior_ = (1.0 - config_.dirichlet_epsilon) * root.children[i]->policy_prior_ +
                                            config_.dirichlet_epsilon * (noise[i] / noise_sum);
    }
}