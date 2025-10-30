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

std::deque<float> convert_position_to_tensor(const Position& pos) {
    std::string fen = pos.fen();
    TensorResult<int> result = fen_to_tensor(fen, 0);
    return result.tensor;
}

// =================================================================
// MCTSNode IMPLEMENTATION
// =================================================================

MCTSNode::MCTSNode(Move move, MCTSNode* parent, float policy_prior)
    : move(move), parent(parent), policy_prior_(policy_prior), virtual_loss_(0) {}

double MCTSNode::q_value() const {
    int v = visits.load(std::memory_order_relaxed);
    int vl = virtual_loss_.load(std::memory_order_relaxed);
    int total_visits = v + vl;
    if (total_visits == 0) {
        return 0.0;
    }
    return total_action_value_.load(std::memory_order_relaxed) / total_visits;
}

MCTSNode* MCTSNode::best_child(double c_puct) const {
    MCTSNode* best = nullptr;
    double max_score = -std::numeric_limits<double>::max();
    int parent_visits = visits.load(std::memory_order_relaxed);
    int parent_virtual_loss = virtual_loss_.load(std::memory_order_relaxed);
    int total_parent_visits = parent_visits + parent_virtual_loss;

    for (const auto& child : children) {
        double q = child->q_value();
        int child_visits = child->visits.load(std::memory_order_relaxed);
        int child_virtual_loss = child->virtual_loss_.load(std::memory_order_relaxed);
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

void MCTSNode::expand(Position& pos, const std::deque<float>& policy_priors) {
    if (pos.turn() == WHITE) {
        MoveList<WHITE> legal_moves(pos);
        for (const auto& m : legal_moves) {
            int policy_index = get_policy_index_for_move(m);
            float prior = (policy_index != -1 && policy_index < policy_priors.size())
                              ? policy_priors[policy_index]
                              : 0.0f;
            children.push_back(std::make_unique<MCTSNode>(m, this, prior));
        }
    } else {
        MoveList<BLACK> legal_moves(pos);
        for (const auto& m : legal_moves) {
            int policy_index = get_policy_index_for_move(m);
            float prior = (policy_index != -1 && policy_index < policy_priors.size())
                              ? policy_priors[policy_index]
                              : 0.0f;
            children.push_back(std::make_unique<MCTSNode>(m, this, prior));
        }
    }
}

void MCTSNode::backpropagate(double value) {
    MCTSNode* node = this;
    while (node != nullptr) {
        node->visits.fetch_add(1, std::memory_order_relaxed);
        double current_value = node->total_action_value_.load(std::memory_order_relaxed);
        while (!node->total_action_value_.compare_exchange_weak(
            current_value, current_value + value, std::memory_order_relaxed))
            ;
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
        std::cout << "[MCTS] Clearing evaluator state..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void MCTS::set_dirichlet_alpha(double alpha) {
    config_.dirichlet_alpha = alpha;
}


Move MCTS::run_search(const Position& initial_pos, int iterations, bool clear_after_search) {
    MCTSNode root(Move(), nullptr, 1.0f);

    // std::cout << "\n[MCTS] Starting fresh search for position: " << initial_pos.fen() << std::endl;
    // std::cout << "[MCTS] Turn: " << (initial_pos.turn() == WHITE ? "WHITE" : "BLACK") << std::endl;

    std::deque<float> root_tensor = convert_position_to_tensor(initial_pos);
    // std::cout << "[MCTS] Evaluating root position..." << std::endl;
    std::future<EvaluationResult> root_future = evaluator_->queue_request(std::move(root_tensor));
    EvaluationResult root_eval = root_future.get();
    // std::cout << "[MCTS] Root evaluation complete. Value: " << root_eval.value << std::endl;

    {
        std::scoped_lock lock(root.expansion_mutex_);
        Position pos_copy = initial_pos;
        root.expand(pos_copy, root_eval.policy);
    }

    root.backpropagate(root_eval.value);
    add_dirichlet_noise(root);
    // std::cout << "[MCTS] Root expanded with " << root.children.size() << " children" << std::endl;

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

    // std::cout << "[MCTS] Search complete. Root visits: " << root.visits.load() << std::endl;
    Move best_move = select_best_move(root);

    if (clear_after_search) {
        clear_evaluator_state();
        // std::cout << "[MCTS] Evaluator state cleared for next search" << std::endl;
    }

    return best_move;
}

void MCTS::search_worker(const Position& root_pos, MCTSNode* root) {
    bool turn_ = root_pos.turn();
    bool root_turn = root_pos.turn();
    Position pos = root_pos;
    MCTSNode* node = root;
    std::deque<MCTSNode*> path;

    while (!node->children.empty()) {
        node->virtual_loss_.fetch_add(1, std::memory_order_relaxed);
        path.push_back(node);
        node = node->best_child(config_.c_puct);

        if (!node) {
            for (auto* path_node : path) {
                path_node->virtual_loss_.fetch_sub(1, std::memory_order_relaxed);
            }
            return;
        }

        if (pos.turn() == WHITE)
            pos.play<WHITE>(node->move);
        else
            pos.play<BLACK>(node->move);
    }

    node->virtual_loss_.fetch_add(1, std::memory_order_relaxed);
    path.push_back(node);

    bool should_expand = false;
    {
        std::scoped_lock lock(node->expansion_mutex_);
        should_expand = node->children.empty();
    }

    if (should_expand) {
        std::deque<float> pos_tensor = convert_position_to_tensor(pos);
        std::future<EvaluationResult> future_eval = evaluator_->queue_request(std::move(pos_tensor));
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        EvaluationResult eval = future_eval.get();

        {
            std::scoped_lock lock(node->expansion_mutex_);
            if (node->children.empty()) {
                node->expand(pos, eval.policy);
            }
        }

        double value = eval.value;
        if (pos.turn() != root_turn) value = -value;
        if (is_checkmate(pos))
            value = (pos.turn() == root_turn) ? -1.0 : 1.0;
        else if (is_stalemate(pos))
            value = 0.0;

        node->backpropagate(value);

        for (auto* path_node : path) {
            path_node->virtual_loss_.fetch_sub(1, std::memory_order_relaxed);
        }
    } else {
        for (auto* path_node : path) {
            path_node->virtual_loss_.fetch_sub(1, std::memory_order_relaxed);
        }
    }
}

Move MCTS::select_best_move(const MCTSNode& root) const {
    if (root.children.empty()) return Move();

    int total_visits = 0;
    for (const auto& child : root.children) {
        total_visits += child->visits.load(std::memory_order_relaxed);
    }

    if (config_.verbose) {
        std::cout << "\n=== Move Selection (Temperature=" << std::fixed << std::setprecision(2)
                  << config_.temperature << ") ===" << std::endl;
    }

    std::vector<double> visit_counts;
    std::vector<MCTSNode*> move_nodes;

    for (const auto& child : root.children) {
        int child_visits = child->visits.load(std::memory_order_relaxed);
        visit_counts.push_back(static_cast<double>(child_visits));
        move_nodes.push_back(child.get());

        if (config_.verbose) {
            std::cout << "Move: " << child->move << " | Visits: " << child_visits
                      << " | Q-value: " << std::fixed << std::setprecision(4) << child->q_value()
                      << std::endl;
        }
    }

    std::vector<double> probabilities = apply_temperature(visit_counts, config_.temperature);
    MCTSNode* selected_node = sample_from_distribution(move_nodes, probabilities);
    if (config_.verbose) {
    std::cout << "\n[MCTS] Root visits: " << root.visits.load(std::memory_order_relaxed)
              << " | Sum of child visits: " << total_visits
              << " | Selected move: " << (selected_node ? selected_node->move : Move()) << std::endl;
    }

    return selected_node ? selected_node->move : Move();
}

std::vector<double> MCTS::apply_temperature(const std::vector<double>& visit_counts,
                                            double temperature) const {
    std::vector<double> probabilities;

    if (temperature < 1e-6) {
        double max_visits = *std::max_element(visit_counts.begin(), visit_counts.end());
        for (double count : visit_counts) {
            probabilities.push_back(count == max_visits ? 1.0 : 0.0);
        }
        double sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
        if (sum > 0) {
            for (auto& p : probabilities) p /= sum;
        }
    } else {
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

    std::deque<double> noise;
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
