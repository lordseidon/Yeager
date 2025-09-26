#include "mcts.h"
#include "types.h"
#include "evaluator.h"
#include <limits>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>

// MCTS Configuration struct for hyperparameters
struct MCTSConfig {
    double c_puct = 2.414;         // PUCT exploration constant (default UCB1 value)
    double temperature = 4.0;      // Temperature for move selection
    double fpu_reduction = 0.25;   // First Play Urgency reduction
    double fpu_value = 0.0;        // First Play Urgency base value
    double noise_alpha = 0.3;      // Dirichlet noise alpha
    double noise_epsilon = 0.25;   // Dirichlet noise mixing ratio
    double value_weight = 1.0;     // Weight for value network
    int min_visits_temperature = 30; // Minimum visits before applying temperature
    bool add_noise = false;        // Whether to add Dirichlet noise to root
    bool verbose = true;           // Verbose output
};

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

// Enhanced best_child with PUCT and FPU support
MCTSNode* MCTSNode::best_child(double c_param) const {
    double best_score = -std::numeric_limits<double>::max();
    MCTSNode* best_child = nullptr;
    
    // FPU parameters - can be adjusted as needed
    const double fpu_value = 0.0;
    const double fpu_reduction = 0.25;

    for (const auto& child : children) {
        double q_value = 0.0;
        double u_value = 0.0;
        
        if (child->visits > 0) {
            // Standard PUCT for visited nodes
            q_value = child->wins / child->visits;
            u_value = c_param * std::sqrt(std::log(visits) / child->visits);
        } else {
            // First Play Urgency for unvisited nodes
            // Assume uniform prior probability
            double uniform_prior = 1.0 / children.size();
            q_value = fpu_value - fpu_reduction * std::sqrt(uniform_prior);
            u_value = c_param * uniform_prior * std::sqrt(visits);
        }
        
        double score = q_value + u_value;
        
        if (score > best_score) {
            best_score = score;
            best_child = child.get();
        }
    }
    return best_child;
}

MCTSNode* MCTSNode::expand(Position& pos) {
    if (pos.turn() == WHITE) {
        MoveList<WHITE> list(pos);

        int move_index = std::rand() % untried_moves.size();
        Move expanding_move = untried_moves[move_index];
        untried_moves.erase(untried_moves.begin() + move_index);

        pos.play<WHITE>(expanding_move);
        children.push_back(std::make_unique<MCTSNode>(pos, expanding_move, this));
    } else {
        MoveList<BLACK> list(pos);

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
    double normalized_eval = (double)eval / 100.0;
    // Clamp to [-1, 1] range
    if (normalized_eval > 1.0) normalized_eval = 1.0;
    if (normalized_eval < -1.0) normalized_eval = -1.0;
    
    return normalized_eval;
}

void MCTSNode::backpropagate(double result) {
    visits++;
    wins += result;
    if (parent) {
        parent->backpropagate(-result);
    }
}

// Helper function to add Dirichlet noise to root node children
void add_dirichlet_noise_to_children(MCTSNode& root, const MCTSConfig& config) {
    if (root.children.empty()) return;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::gamma_distribution<double> gamma(config.noise_alpha, 1.0);
    
    std::vector<double> noise;
    double noise_sum = 0.0;
    
    for (size_t i = 0; i < root.children.size(); ++i) {
        double n = gamma(gen);
        noise.push_back(n);
        noise_sum += n;
    }
    
    // Normalize noise - in this simplified version, we'll just use it to bias selection
    for (double& n : noise) {
        n /= noise_sum;
    }
}

// Temperature-based move selection
Move select_move_with_temperature(const std::vector<std::unique_ptr<MCTSNode>>& children, 
                                  const MCTSConfig& config) {
    if (children.empty()) return Move();
    
    std::vector<double> visit_counts;
    double max_visits = 0.0;
    
    for (const auto& child : children) {
        double visits = static_cast<double>(child->visits);
        visit_counts.push_back(visits);
        max_visits = std::max(max_visits, visits);
    }
    
    if (config.temperature == 0.0 || max_visits < config.min_visits_temperature) {
        // Select move with highest visit count
        auto max_it = std::max_element(visit_counts.begin(), visit_counts.end());
        int best_index = std::distance(visit_counts.begin(), max_it);
        return children[best_index]->move;
    }
    
    // Apply temperature
    std::vector<double> probabilities;
    double prob_sum = 0.0;
    
    for (double visits : visit_counts) {
        double prob = std::pow(visits, 1.0 / config.temperature);
        probabilities.push_back(prob);
        prob_sum += prob;
    }
    
    // Normalize probabilities
    for (double& prob : probabilities) {
        prob /= prob_sum;
    }
    
    // Sample from distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
    
    int selected_index = dist(gen);
    return children[selected_index]->move;
}

// Enhanced MCTS search with configuration
Move mcts_search_with_config(Position& initial_pos, int iterations, const MCTSConfig& config) {
    if (config.verbose) {
        std::cout << "Starting Enhanced MCTS search..." << std::endl;
        std::cout << "Config: c_puct=" << config.c_puct 
                  << ", temp=" << config.temperature 
                  << ", fpu_reduction=" << config.fpu_reduction << std::endl;
    }
    
    MCTSNode root(initial_pos);
    
    // FPU values from config (for display purposes)
    const double fpu_value = config.fpu_value;
    const double fpu_reduction = config.fpu_reduction;

    for (int i = 0; i < iterations; ++i) {
        Position pos = initial_pos;
        MCTSNode* node = &root;

        // Selection
        while (node->is_fully_expanded() && !node->children.empty()) {
            node = node->best_child(config.c_puct);
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

        // Simulation with value weighting
        double result = node->rollout(pos) * config.value_weight;

        // Backpropagation
        node->backpropagate(result);
        
        // Add noise on first few iterations if enabled
        if (i < 5 && config.add_noise) {
            add_dirichlet_noise_to_children(root, config);
        }
    }

    // Show top ranking moves
    if (config.verbose) {
        std::cout << "\n=== Enhanced MCTS MOVE RANKINGS ===" << std::endl;
        std::vector<MCTSNode*> child_ptrs;
        for (const auto& child : root.children) {
            child_ptrs.push_back(child.get());
        }
        
        // Sort by visit count (descending)
        std::sort(child_ptrs.begin(), child_ptrs.end(), 
            [](const MCTSNode* a, const MCTSNode* b) {
                return a->visits > b->visits;
            });
        
        // Display top moves
        for (size_t i = 0; i < child_ptrs.size() && i < 10; ++i) {
            const MCTSNode* child = child_ptrs[i];
            double win_rate = (child->visits > 0) ? child->get_wins() / child->visits : 0.0;
            double puct_score = 0.0;
            
            if (child->visits > 0) {
                puct_score = win_rate + config.c_puct * std::sqrt(std::log(root.visits) / child->visits);
            } else {
                double uniform_prior = 1.0 / child_ptrs.size();
                puct_score = fpu_value - fpu_reduction * std::sqrt(uniform_prior) + 
                           config.c_puct * uniform_prior * std::sqrt(root.visits);
            }
            
            std::cout << (i+1) << ". Move: " << child->move 
                      << " | Visits: " << child->visits 
                      << " | Win Rate: " << std::fixed << std::setprecision(4) << win_rate
                      << " | PUCT Score: " << std::fixed << std::setprecision(4) << puct_score
                      << " | Score: " << std::fixed << std::setprecision(2) << child->get_wins()
                      << std::endl;
        }
        std::cout << "=====================================" << std::endl;
    }

    // Select final move using temperature
    return select_move_with_temperature(root.children, config);
}

// Original function with default parameters for backward compatibility
Move mcts_search(Position& initial_pos, int iterations) {
    MCTSConfig default_config;
    default_config.c_puct = 1.414;  // Keep original UCB1 constant
    default_config.temperature = 0.5; // Deterministic selection (pick best)
    default_config.verbose = true;
    default_config.add_noise = false;
    default_config.value_weight = 1.0;
    
    return mcts_search_with_config(initial_pos, iterations, default_config);
}

// Additional utility functions for easy customization

// Tournament play configuration (deterministic, no exploration)
Move mcts_search_tournament(Position& initial_pos, int iterations) {
    MCTSConfig config;
    config.c_puct = 1.0;           // Lower exploration
    config.temperature = 0.0;      // Always pick best move
    config.add_noise = false;      // No exploration noise
    config.verbose = false;        // Quiet mode
    config.value_weight = 1.0;
    
    return mcts_search_with_config(initial_pos, iterations, config);
}

// Training/self-play configuration (higher exploration, temperature)
Move mcts_search_training(Position& initial_pos, int iterations) {
    MCTSConfig config;
    config.c_puct = 2.0;           // Higher exploration
    config.temperature = 1.2;      // More random move selection
    config.add_noise = true;       // Add exploration noise
    config.verbose = true;
    config.value_weight = 1.0;
    config.noise_alpha = 0.3;
    config.noise_epsilon = 0.25;
    
    return mcts_search_with_config(initial_pos, iterations, config);
}

// Analysis configuration (balanced exploration, detailed output)
Move mcts_search_analysis(Position& initial_pos, int iterations, double c_puct = 2.414, 
                         double temperature = 2.5) {
    MCTSConfig config;
    config.c_puct = c_puct;
    config.temperature = temperature;
    config.add_noise = false;
    config.verbose = true;
    config.value_weight = 1.0;
    config.fpu_reduction = 0.25;
    config.min_visits_temperature = 50;
    
    return mcts_search_with_config(initial_pos, iterations, config);
}