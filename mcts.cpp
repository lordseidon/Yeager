#include "mcts.h"
#include "evaluator.h"
#include "position_tensor.h"
#include "fen_to_tensor.h"
#include <cmath>
#include <limits>
#include <deque>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <thread>

// =================================================================
// PLACEHOLDER FUNCTIONS - YOU MUST IMPLEMENT THESE
// =================================================================

// Maps a move object to an integer index for the policy deque (0 to 4479)
// This is a CRITICAL function you must design based on your move encoding.
int get_policy_index_for_move(const Move& move) {
    return MoveMappings::get_policy_index_for_move(move);
}

// Converts your board `Position` object into a flattened 1D deque of floats
// with the shape (8*8*18 = 1152) that your neural network expects.
std::deque<float> convert_position_to_tensor(const Position& pos) {
    // Convert Position to FEN string
    std::string fen = pos.fen();  // Using the fen() method from Position class
    
    // Use a dummy ID since we don't need it for MCTS
    TensorResult<int> result = fen_to_tensor(fen, 0);
    // std::cout<<"FEN: "<<fen;
    
    return result.tensor;
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

// The function signature is reverted to take a non-const reference.
void MCTSNode::expand(Position& pos, const std::deque<float>& policy_priors) {
    // std::cout << "\n=== EXPAND DEBUG ===" << std::endl;
    // std::cout << "Turn: " << (pos.turn() == WHITE ? "WHITE" : "BLACK") << std::endl;
    // std::cout << "Policy priors size: " << policy_priors.size() << std::endl;
    
    if (pos.turn() == WHITE) {
        MoveList<WHITE> legal_moves(pos);
        // std::cout << "Legal moves count: " << legal_moves.size() << std::endl;
        
        for (const auto& m : legal_moves) {
            int policy_index = get_policy_index_for_move(m);
            float prior = (policy_index != -1 && policy_index < policy_priors.size()) 
                         ? policy_priors[policy_index] : 0.0f;
            
            // std::cout << "Move: " << m 
            //           << " -> Index: " << policy_index 
            //           << " -> Prior: " << prior << std::endl;
            
            children.push_back(std::make_unique<MCTSNode>(m, this, prior));
        }
    } else {
        MoveList<BLACK> legal_moves(pos);
        // std::cout << "Legal moves count: " << legal_moves.size() << std::endl;
        
        for (const auto& m : legal_moves) {
            int policy_index = get_policy_index_for_move(m);
            float prior = (policy_index != -1 && policy_index < policy_priors.size()) 
                         ? policy_priors[policy_index] : 0.0f;
            
            // std::cout << "Move: " << m 
            //           << " -> Index: " << policy_index 
            //           << " -> Prior: " << prior << std::endl;
            
            children.push_back(std::make_unique<MCTSNode>(m, this, prior));
        }
    }
    
    // std::cout << "Total children created: " << children.size() << std::endl;
    // std::cout << "==================\n" << std::endl;
}

void MCTSNode::backpropagate(double value) {
    MCTSNode* node = this;
    while (node != nullptr) {
        node->visits.fetch_add(1, std::memory_order_relaxed);
        
        // Correctly perform an atomic add on a double using a compare-exchange loop.
        double current_value = node->total_action_value_.load(std::memory_order_relaxed);
        while (!node->total_action_value_.compare_exchange_weak(current_value, current_value + value, std::memory_order_relaxed));

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
    std::deque<float> root_tensor = convert_position_to_tensor(initial_pos);
    std::cout << "[CLIENT] Main thread: Queueing initial root evaluation..." << std::endl;
    EvaluationResult root_eval = evaluator_->queue_request(std::move(root_tensor)).get();
    std::cout << "[CLIENT] Main thread: Initial root evaluation received." << std::endl;

    { // Lock is not strictly needed for root, but good practice
        std::scoped_lock lock(root.expansion_mutex_);
        Position pos_copy = initial_pos; 
        
        // Apply softmax to root evaluation
        std::deque<float> softmax_policy(root_eval.policy.size());
        float max_val = *std::max_element(root_eval.policy.begin(), root_eval.policy.end());
        float sum = 0.0f;
        for (size_t i = 0; i < root_eval.policy.size(); i++) {
            softmax_policy[i] = std::exp(root_eval.policy[i] - max_val);
            sum += softmax_policy[i];
        }
        for (size_t i = 0; i < root_eval.policy.size(); i++) {
            softmax_policy[i] /= sum;
        }
        root.expand(pos_copy, softmax_policy);
    }
    
    // Backpropagate root value directly (no flip needed for root)
    root.backpropagate(root_eval.value);

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
    static std::atomic<int> iteration_count{0};
    Position pos = root_pos;
    MCTSNode* node = root;
    
    // Virtual loss to prevent threads from taking same path
    const int VIRTUAL_LOSS = 4;
    std::deque<MCTSNode*> path;

    // 1. SELECTION: Traverse the tree using PUCT with virtual loss
    while (!node->children.empty()) {
        node = node->best_child(config_.c_puct);
        path.push_back(node);
        
        // Apply virtual loss to make this node temporarily less attractive
        node->visits.fetch_add(VIRTUAL_LOSS, std::memory_order_relaxed);
        
        if (pos.turn() == WHITE) pos.play<WHITE>(node->move);
        else pos.play<BLACK>(node->move);
    }

    // Check for a terminal game state
    if (is_checkmate(pos) || is_stalemate(pos)) {
        double result = (is_checkmate(pos)) ? -1.0 : 0.0;
        
        // Remove virtual loss before backpropagation
        for (auto n : path) {
            n->visits.fetch_sub(VIRTUAL_LOSS, std::memory_order_relaxed);
        }
        
        node->backpropagate(result);
        
        int count = iteration_count.fetch_add(1);
        if (count % 500 == 0) {
            std::cout << "Completed iterations: " << count << std::endl;
        }
        return;
    }

    // 2. EXPANSION & EVALUATION
    std::scoped_lock lock(node->expansion_mutex_);
    
    if (node->children.empty()) {
        std::deque<float> pos_tensor = convert_position_to_tensor(pos);
        std::future<EvaluationResult> future_eval = evaluator_->queue_request(std::move(pos_tensor));
        EvaluationResult eval = future_eval.get();

        // Apply softmax to policy
        std::deque<float> softmax_policy(eval.policy.size());
        float max_val = *std::max_element(eval.policy.begin(), eval.policy.end());
        float sum = 0.0f;
        for (size_t i = 0; i < eval.policy.size(); i++) {
            softmax_policy[i] = std::exp(eval.policy[i] - max_val);
            sum += softmax_policy[i];
        }
        for (size_t i = 0; i < eval.policy.size(); i++) {
            softmax_policy[i] /= sum;
        }
        
        node->expand(pos, softmax_policy);
        
        // Remove virtual loss before backpropagation
        for (auto n : path) {
            n->visits.fetch_sub(VIRTUAL_LOSS, std::memory_order_relaxed);
        }
        
        // Flip value based on perspective
        double value = (pos.turn() == WHITE) ? eval.value : -eval.value;
        node->backpropagate(value);
    } else {
        // Another thread already expanded this node, just remove virtual loss
        for (auto n : path) {
            n->visits.fetch_sub(VIRTUAL_LOSS, std::memory_order_relaxed);
        }
    }
    
    int count = iteration_count.fetch_add(1);
    if (count % 500 == 0) {
        std::cout << "Completed iterations: " << count << std::endl;
    }
}


Move MCTS::select_best_move(const MCTSNode& root) const {
    if (root.children.empty()) return Move(); // Should not happen in a real search

    MCTSNode* best_move_node = nullptr;
    int max_visits = -1;

    // In tournament play or late in the game, you'd pick the move with the most visits (temp=0).
    // For training, you can sample from a distribution based on visit counts.
    for (const auto& child : root.children) {
        int child_visits = child->visits.load(std::memory_order_relaxed);
        if (child_visits > max_visits) {
            max_visits = child_visits;
            best_move_node = child.get();
        }
        if (config_.verbose) {
            std::cout << "Move: " << child->move 
                      << " | Visits: " << child_visits
                      << " | Win Rate: " << std::fixed << std::setprecision(4) << child->q_value()
                      << std::endl;
        }
    }
    return best_move_node ? best_move_node->move : Move();
}

void MCTS::add_dirichlet_noise(MCTSNode& root) {
    if (config_.dirichlet_epsilon == 0.0 || root.children.empty()) return;

    std::deque<double> noise;
    std::random_device rd;
    // Corrected the typo from mt1937 to mt19937.
    std::mt19937 gen(rd());
    std::gamma_distribution<double> gamma(config_.dirichlet_alpha, 1.0);
    
    double noise_sum = 0.0;
    for (size_t i = 0; i < root.children.size(); ++i) {
        double n = gamma(gen);
        noise.push_back(n);
        noise_sum += n;
    }

    if (noise_sum < 1e-6) return; // Avoid division by zero
    
    for (size_t i = 0; i < root.children.size(); ++i) {
        root.children[i]->policy_prior_ = (1.0 - config_.dirichlet_epsilon) * root.children[i]->policy_prior_ +
                                            config_.dirichlet_epsilon * (noise[i] / noise_sum);
    }
}