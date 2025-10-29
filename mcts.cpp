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

int get_policy_index_for_move(const Move& move) {
    return MoveMappings::get_policy_index_for_move(move);
}

std::vector<float> convert_position_to_tensor(const Position& pos, const std::deque<std::string>& fen_history) {
    std::deque<std::string> full_history = fen_history;
    full_history.push_back(pos.fen());
    TensorResult<int> result = fen_history_to_tensor(full_history, 0);
    
    // Convert deque to vector
    return std::vector<float>(result.tensor.begin(), result.tensor.end());
}

// Softmax function for policy logits
std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());
    
    // Find max for numerical stability
    float max_logit = *std::max_element(logits.begin(), logits.end());
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    
    // Normalize
    if (sum > 0.0f) {
        for (auto& p : probs) {
            p /= sum;
        }
    }
    
    return probs;
}

// =================================================================
// MCTSNode IMPLEMENTATION
// =================================================================

MCTSNode::MCTSNode(Move move, MCTSNode* parent, float policy_prior)
    : move(move), parent(parent), policy_prior_(policy_prior), virtual_loss_(0) {}

double MCTSNode::q_value() const {
    int v = visits.load(std::memory_order_acquire);
    int vl = virtual_loss_.load(std::memory_order_acquire);
    int total_visits = v + vl;
    if (total_visits == 0) {
        return 0.0;
    }
    return total_action_value_.load(std::memory_order_acquire) / total_visits;
}

MCTSNode* MCTSNode::best_child(double c_puct) const {
    MCTSNode* best = nullptr;
    double max_score = -std::numeric_limits<double>::max();
    int parent_visits = visits.load(std::memory_order_acquire);
    int parent_virtual_loss = virtual_loss_.load(std::memory_order_acquire);
    int total_parent_visits = parent_visits + parent_virtual_loss;

    for (const auto& child : children) {
        double q = child->q_value();
        int child_visits = child->visits.load(std::memory_order_acquire);
        int child_virtual_loss = child->virtual_loss_.load(std::memory_order_acquire);
        int total_child_visits = child_visits + child_virtual_loss;

        double u = c_puct * child->policy_prior_ *
                   (std::sqrt(total_parent_visits) / (1 + total_child_visits));
        double score = q + u;

        if (score > max_score) {
            max_score = score;
            best = child.get();
        }
    }
    return best;
}

void MCTSNode::expand(Position& pos, const std::vector<float>& policy_priors) {
    // Apply softmax to policy priors
    std::vector<float> policy_probs = softmax(policy_priors);
    
    if (pos.turn() == WHITE) {
        MoveList<WHITE> legal_moves(pos);
        for (const auto& m : legal_moves) {
            int policy_index = get_policy_index_for_move(m);
            float prior = (policy_index != -1 && policy_index < static_cast<int>(policy_probs.size()))
                              ? policy_probs[policy_index]
                              : 0.0f;
            children.push_back(std::make_unique<MCTSNode>(m, this, prior));
        }
    } else {
        MoveList<BLACK> legal_moves(pos);
        for (const auto& m : legal_moves) {
            int policy_index = get_policy_index_for_move(m);
            float prior = (policy_index != -1 && policy_index < static_cast<int>(policy_probs.size()))
                              ? policy_probs[policy_index]
                              : 0.0f;
            children.push_back(std::make_unique<MCTSNode>(m, this, prior));
        }
    }
}

void MCTSNode::backpropagate(double value) {
    MCTSNode* node = this;
    while (node != nullptr) {
        // Increment visits with release ordering so other threads see the update
        node->visits.fetch_add(1, std::memory_order_release);
        
        // Update total action value with proper CAS loop
        double current_value = node->total_action_value_.load(std::memory_order_acquire);
        while (!node->total_action_value_.compare_exchange_weak(
            current_value, 
            current_value + value,
            std::memory_order_release,   // Success ordering
            std::memory_order_acquire))  // Failure ordering (reloads current_value)
        {
            // CAS failed, loop continues with updated current_value
        }
        value = -value;
        
        node = node->parent;
    }
}

// =================================================================
// MCTS IMPLEMENTATION
// =================================================================

MCTS::MCTS(MCTSConfig config, std::unique_ptr<RemoteEvaluator> evaluator)
    : config_(std::move(config)), evaluator_(std::move(evaluator)) {}

void MCTS::clear_evaluator_state() {
    if (evaluator_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(0));
    }
}

void MCTS::set_dirichlet_alpha(double alpha) {
    config_.dirichlet_alpha = alpha;
}

Move MCTS::run_search(const Position& initial_pos, int iterations, bool clear_after_search) {
    return run_search(initial_pos, iterations, clear_after_search, {});
}

Move MCTS::run_search(const Position& initial_pos, int iterations, bool clear_after_search, const std::deque<std::string>& position_history) {
    MCTSNode root(Move(), nullptr, 1.0f);
    root.fen_history = position_history;

    // Evaluate and expand root
    std::vector<float> root_tensor = convert_position_to_tensor(initial_pos, position_history);
    std::future<EvaluationResult> root_future = evaluator_->queue_request(std::move(root_tensor));
    EvaluationResult root_eval = root_future.get();

    {
        std::scoped_lock lock(root.expansion_mutex_);
        Position pos_copy = initial_pos;
        root.expand(pos_copy, root_eval.policy);
    }

    // Initial backpropagation for root
    root.backpropagate(root_eval.value);
    
    // Add exploration noise
    add_dirichlet_noise(root);

    // Launch worker threads
    std::vector<std::thread> threads;
    threads.reserve(config_.num_threads);
    std::atomic<int> iterations_remaining(iterations);

    for (int i = 0; i < config_.num_threads; ++i) {
        threads.emplace_back([this, &initial_pos, &root, &iterations_remaining]() {
            while (true) {
                int remaining = iterations_remaining.fetch_sub(1, std::memory_order_relaxed);
                if (remaining <= 0) break;
                this->search_worker(initial_pos, &root);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    Move best_move = select_best_move(root);

    if (clear_after_search) {
        clear_evaluator_state();
    }

    return best_move;
}

void MCTS::search_worker(const Position& root_pos, MCTSNode* root) {
    bool root_turn = root_pos.turn();
    
    Position pos = root_pos;
    MCTSNode* node = root;
    std::vector<MCTSNode*> path;
    std::deque<std::string> current_fen_history = root->fen_history;

    // =================================================================
    // PHASE 1: SELECTION - Traverse tree with virtual loss
    // =================================================================
    while (!node->children.empty()) {
        // Add virtual loss to discourage other threads
        node->virtual_loss_.fetch_add(1, std::memory_order_relaxed);
        path.push_back(node);
        
        // Select best child
        node = node->best_child(config_.c_puct);

        if (!node) {
            // Selection failed - remove virtual loss and return
            for (auto* path_node : path) {
                path_node->virtual_loss_.fetch_sub(1, std::memory_order_relaxed);
            }
            return;
        }

        // Apply move
        std::string current_fen = pos.fen();
        
        if (pos.turn() == WHITE)
            pos.play<WHITE>(node->move);
        else
            pos.play<BLACK>(node->move);
        
        current_fen_history.push_back(current_fen);
        if (current_fen_history.size() > 7) {
            current_fen_history.pop_front();
        }
    }

    // Add virtual loss to leaf node
    node->virtual_loss_.fetch_add(1, std::memory_order_relaxed);
    path.push_back(node);

    // =================================================================
    // PHASE 2: EXPANSION - Try to expand the leaf node
    // =================================================================
    bool should_expand = false;
    {
        std::scoped_lock lock(node->expansion_mutex_);
        
        // Double-check children are empty while holding lock
        should_expand = node->children.empty();
        
        if (!should_expand) {
            // Another thread expanded while we were waiting
            // Remove virtual loss and return
            for (auto* path_node : path) {
                path_node->virtual_loss_.fetch_sub(1, std::memory_order_relaxed);
            }
            return;
        }
        
        // We have exclusive expansion rights - proceed with evaluation
    }
    // Lock is released here, but we still own expansion rights

    // =================================================================
    // PHASE 3: EVALUATION - Get neural network evaluation
    // =================================================================
    std::vector<float> pos_tensor = convert_position_to_tensor(pos, current_fen_history);
    std::future<EvaluationResult> future_eval = evaluator_->queue_request(std::move(pos_tensor));
    
    // Wait for neural network evaluation (blocking, but batching happens here)
    EvaluationResult eval = future_eval.get();
    
    // =================================================================
    // PHASE 4: EXPANSION - Create child nodes
    // =================================================================
    {
        std::scoped_lock lock(node->expansion_mutex_);
        // Final check before expanding
        if (node->children.empty()) {
            node->fen_history = current_fen_history;
            node->expand(pos, eval.policy);
        }
    }

    // =================================================================
    // PHASE 5: BACKPROPAGATION - Update tree with evaluation
    // =================================================================
    // double value = eval.value;
    
    // // Flip value if needed (from leaf perspective to root perspective)
    // if (pos.turn() != root_turn) {
    //     value = -value;
    // }
    
    // // Check for terminal conditions
    // if (is_checkmate(pos)) {
    //     value = (pos.turn() == root_turn) ? -1.0 : 1.0;
    // } else if (is_stalemate(pos)) {
    //     value = 0.0;
    // }
    
    // // Backpropagate value up the tree
    // node->backpropagate(value);

    double value = eval.value;
    
    // Flip value if needed (from leaf perspective to root perspective)
    if (pos.turn() != root_turn) {
        value = -value;
    }
    
    // Check for terminal conditions
    if (is_checkmate(pos)) {
        value = (pos.turn() == root_turn) ? -1.0 : 1.0;
    } else if (is_stalemate(pos)) {
        value = 0.0;
    }
    
    // Backpropagate value up the tree
    node->backpropagate(value);

    // =================================================================
    // PHASE 6: CLEANUP - Remove virtual loss AFTER backpropagation
    // =================================================================
    // CRITICAL: This must happen AFTER backpropagation completes
    // so that visit counts are updated before we remove virtual loss
    for (auto* path_node : path) {
        path_node->virtual_loss_.fetch_sub(1, std::memory_order_relaxed);
    }
}

Move MCTS::select_best_move(const MCTSNode& root) const {
    if (root.children.empty()) return Move();

    // Calculate total visits for logging
    int total_visits = 0;
    for (const auto& child : root.children) {
        total_visits += child->visits.load(std::memory_order_acquire);
    }

    if (config_.verbose) {
        std::cout << "\n=== Move Selection (Temperature=" << std::fixed << std::setprecision(2)
                  << config_.temperature << ") ===" << std::endl;
    }

    std::vector<double> visit_counts;
    std::vector<MCTSNode*> move_nodes;

    for (const auto& child : root.children) {
        int child_visits = child->visits.load(std::memory_order_acquire);
        visit_counts.push_back(static_cast<double>(child_visits));
        move_nodes.push_back(child.get());

        if (config_.verbose) {
            std::cout << "Move: " << child->move << " | Visits: " << child_visits
                      << " | Q-value: " << std::fixed << std::setprecision(4) << child->q_value()
                      << " | Prior: " << std::fixed << std::setprecision(4) << child->policy_prior_
                      << std::endl;
        }
    }

    // Temperature = 0: deterministic selection (most visits)
    if (config_.temperature < 1e-6) {
        auto max_it = std::max_element(visit_counts.begin(), visit_counts.end());
        size_t best_idx = std::distance(visit_counts.begin(), max_it);
        
        if (config_.verbose) {
            std::cout << "\n[MCTS] Deterministic selection (temp=0): " 
                      << move_nodes[best_idx]->move << std::endl;
        }
        
        return move_nodes[best_idx]->move;
    }

    // Temperature > 0: probabilistic sampling
    std::vector<double> probabilities = apply_temperature(visit_counts, config_.temperature);
    MCTSNode* selected_node = sample_from_distribution(move_nodes, probabilities);
    
    if (config_.verbose) {
        std::cout << "\n[MCTS] Root visits: " << root.visits.load(std::memory_order_acquire)
                  << " | Sum of child visits: " << total_visits
                  << " | Selected move: " << (selected_node ? selected_node->move : Move()) << std::endl;
    }

    return selected_node ? selected_node->move : Move();
}

std::vector<double> MCTS::apply_temperature(const std::vector<double>& visit_counts,
                                            double temperature) const {
    std::vector<double> probabilities;

    if (temperature < 1e-6) {
        // Should not reach here, but handle it anyway
        double max_visits = *std::max_element(visit_counts.begin(), visit_counts.end());
        for (double count : visit_counts) {
            probabilities.push_back(count == max_visits ? 1.0 : 0.0);
        }
        double sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
        if (sum > 0) {
            for (auto& p : probabilities) p /= sum;
        }
    } else {
        // Apply temperature scaling with numerical stability
        double max_count = *std::max_element(visit_counts.begin(), visit_counts.end());
        double sum = 0.0;
        for (double count : visit_counts) {
            double scaled = std::exp((count - max_count) / temperature);
            probabilities.push_back(scaled);
            sum += scaled;
        }
        if (sum > 0) {
            for (auto& p : probabilities) p /= sum;
        }
    }

    return probabilities;
}

MCTSNode* MCTS::sample_from_distribution(const std::vector<MCTSNode*>& nodes,
                                         const std::vector<double>& probabilities) const {
    if (nodes.empty()) return nullptr;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<int> dist(probabilities.begin(), probabilities.end());
    int selected_index = dist(gen);
    return nodes[selected_index];
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

    if (noise_sum < 1e-6) return;

    for (size_t i = 0; i < root.children.size(); ++i) {
        root.children[i]->policy_prior_ =
            (1.0 - config_.dirichlet_epsilon) * root.children[i]->policy_prior_ +
            config_.dirichlet_epsilon * (noise[i] / noise_sum);
    }
}