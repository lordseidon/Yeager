#include "position.h"
#include "mcts.h"
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <sstream>
#include <random>
#include <filesystem>
#include <map>
#include <vector>
#include <iomanip>
#include <thread>
#include <future>
#include <atomic>
#include <mutex>
#include "remote_evaluator.h"
#include "fen_to_tensor.h"
#include "position_tensor.h"

namespace fs = std::filesystem;

std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");
    return ss.str();
}

std::string get_random_suffix(int digits = 6) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 999999);
    
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(digits) << dis(gen);
    return ss.str();
}



std::string move_to_algebraic(const Move& move, const Position& pos) {
    std::stringstream ss;
    ss << move;
    std::string move_str = ss.str();
    
    size_t capture_pos = move_str.find(" (capture)");
    if (capture_pos != std::string::npos) {
        move_str = move_str.substr(0, capture_pos);
    }
    
    return move_str;
}

std::string build_pgn(const std::vector<Move>& moves, const std::string& result) {
    std::stringstream pgn;
    
    pgn << "[Event \"MCTS Self-Play Game\"]\n";
    pgn << "[Site \"Local\"]\n";
    pgn << "[Date \"" << get_timestamp() << "\"]\n";
    pgn << "[Round \"1\"]\n";
    pgn << "[White \"MCTS Engine\"]\n";
    pgn << "[Black \"MCTS Engine\"]\n";
    pgn << "[Result \"" << result << "\"]\n\n";
    
    int move_number = 1;
    for (size_t i = 0; i < moves.size(); ++i) {
        if (i % 2 == 0) {
            pgn << move_number << ". ";
            ++move_number;
        }
        
        std::stringstream ss;
        ss << moves[i];
        std::string move_str = ss.str();
        
        // Remove " (capture)" suffix if present
        size_t capture_pos = move_str.find(" (capture)");
        if (capture_pos != std::string::npos) {
            move_str = move_str.substr(0, capture_pos);
        }
        
        // Remove castling notation (O-O or O-O-O) if present
        size_t castle_short = move_str.find(" O-O");
        if (castle_short != std::string::npos) {
            move_str = move_str.substr(0, castle_short);
        }
        
        pgn << move_str << " ";
        
        if ((i + 1) % 20 == 0) {
            pgn << "\n";
        }
    }
    
    pgn << result << "\n";
    return pgn.str();
}

bool check_threefold_repetition(const std::map<uint64_t, int>& position_history, const Position& current_pos) {
    uint64_t hash = current_pos.get_hash();
    auto it = position_history.find(hash);
    return (it != position_history.end() && it->second >= 3);
}

// Function to play a single game
struct GameResult {
    std::vector<Move> moves;
    std::string result;
    std::string pgn;
    int move_count;
    std::string starting_fen;
    int game_id;
};

// Thread-safe progress tracking
struct ProgressTracker {
    std::atomic<int> completed_games{0};
    std::atomic<int> total_moves{0};
    std::atomic<int> total_moves_played{0};  // Real-time move counter across all games
    std::atomic<int> white_wins{0};
    std::atomic<int> black_wins{0};
    std::atomic<int> draws{0};
    std::mutex results_mutex;
    std::vector<GameResult> all_results;
    
    void add_result(const GameResult& result) {
        completed_games.fetch_add(1);
        total_moves.fetch_add(result.move_count);
        
        if (result.result == "1-0") {
            white_wins.fetch_add(1);
        } else if (result.result == "0-1") {
            black_wins.fetch_add(1);
        } else if (result.result == "1/2-1/2") {
            draws.fetch_add(1);
        }
        
        std::lock_guard<std::mutex> lock(results_mutex);
        all_results.push_back(result);
    }
};

GameResult play_single_game(MCTS& mcts_engine, int game_id, const MCTSConfig& config, ProgressTracker& progress, bool verbose = false) {
    // Always start from the standard chess starting position
    Position pos;
    std::string starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    Position::set(starting_fen, pos);
    
    std::vector<Move> moves_played;
    std::string game_result = "*";
    
    std::map<uint64_t, int> position_history;
    position_history[pos.get_hash()] = 1;
    
    int full_move_count = 0;
    int iteration_ = 400;
    const int MAX_MOVES = 200;
    
    if (verbose) {
        std::cout << "\n=== GAME " << game_id << " ===" << std::endl;
        std::cout << "Starting FEN: " << starting_fen << std::endl;
    }
    
    while (full_move_count < MAX_MOVES) {
        bool is_white = (pos.turn() == WHITE);
        
        // Dynamically adjust Dirichlet alpha based on move count
        // For first 20 moves (opening), use higher exploration (0.45)
        // After 20 moves, use normal value (0.15)
        if (full_move_count < 20) {
            mcts_engine.set_dirichlet_alpha(0.65);
        } else {
            mcts_engine.set_dirichlet_alpha(0.10);
        }
        
        // Run search with the shared MCTS engine (don't clear after search)
        Move best_move = mcts_engine.run_search(pos, iteration_, false);
        
        if (best_move == Move()) {
            if (!is_white) {
                game_result = "0-1";
            } else {
                game_result = "1-0";
            }
            break;
        }
        
        moves_played.push_back(best_move);
        
        if (is_white) {
            pos.play<WHITE>(best_move);
        } else {
            pos.play<BLACK>(best_move);
        }
        
        full_move_count++;
        
        // Increment the global real-time move counter
        progress.total_moves_played.fetch_add(1);
        
        uint64_t pos_hash = pos.get_hash();
        position_history[pos_hash]++;
        
        bool is_checkmate = false;
        bool is_stalemate = false;
        bool is_threefold_rep = check_threefold_repetition(position_history, pos);
        
        if (pos.turn() == WHITE) {
            MoveList<WHITE> legal_moves(pos);
            if (legal_moves.size() == 0) {
                is_checkmate = pos.in_check<WHITE>();
                is_stalemate = !is_checkmate;
            }
        } else {
            MoveList<BLACK> legal_moves(pos);
            if (legal_moves.size() == 0) {
                is_checkmate = pos.in_check<BLACK>();
                is_stalemate = !is_checkmate;
            }
        }
        
        if (is_checkmate) {
            game_result = !is_white ? "0-1" : "1-0";
            break;
        }
        
        if (is_stalemate || is_threefold_rep) {
            game_result = "1/2-1/2";
            break;
        }
    }
    
    if (full_move_count >= MAX_MOVES) {
        game_result = "1/2-1/2";
    }
    
    std::string pgn = build_pgn(moves_played, game_result);
    
    GameResult result;
    result.moves = moves_played;
    result.result = game_result;
    result.pgn = pgn;
    result.move_count = full_move_count;
    result.starting_fen = starting_fen;
    result.game_id = game_id;
    
    return result;
}

int main() {
    initialise_all_databases();
    zobrist::initialise_zobrist_keys();
    MoveMappings::initialize_move_mappings();
    
    std::string pgn_folder = "pgns";
    if (!fs::exists(pgn_folder)) {
        try {
            fs::create_directory(pgn_folder);
            std::cout << "Created folder: " << pgn_folder << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error creating folder: " << e.what() << std::endl;
            return 1;
        }
    }
    

    
    MCTSConfig config;
    config.num_threads = 16;
    config.batch_size = 256;
    config.c_puct = 3;
    config.temperature = 1.5;
    config.verbose = false;  // Disable verbose for batch processing
    config.dirichlet_alpha = 0.10;
    config.dirichlet_epsilon = 0.1;
    
    std::string server_address = "localhost:50055";
    const int NUM_GAMES = 10;
    
    std::cout << "\n=== MCTS Self-Play PARALLEL Engine ===" << std::endl;
    std::cout << "Server address: " << server_address << std::endl;
    std::cout << "Number of simultaneous games: " << NUM_GAMES << std::endl;
    std::cout << "Using SINGLE shared evaluator for ALL PARALLEL games" << std::endl;
    std::cout << "Expected maximum batch utilization with " << NUM_GAMES << " parallel games!" << std::endl;
    std::cout << "Dirichlet noise: alpha=0.45 (first 20 moves) -> 0.15 (after), epsilon=" 
              << config.dirichlet_epsilon << "\n" << std::endl;
    
    // CREATE A SINGLE EVALUATOR FOR ALL PARALLEL GAMES
    std::cout << "Creating shared evaluator..." << std::endl;
    auto evaluator = std::make_unique<RemoteEvaluator>(server_address, config.batch_size);
    
    // CREATE A SINGLE MCTS ENGINE FOR ALL PARALLEL GAMES
    std::cout << "Creating shared MCTS engine..." << std::endl;
    MCTS mcts_engine(config, std::move(evaluator));
    
    // Progress tracker for thread-safe statistics
    ProgressTracker progress;
    
    auto batch_start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "Starting " << NUM_GAMES << " parallel games..." << std::endl;
    
    // Launch all games in parallel
    std::vector<std::future<GameResult>> game_futures;
    game_futures.reserve(NUM_GAMES);
    
    for (int game_id = 1; game_id <= NUM_GAMES; ++game_id) {
        game_futures.emplace_back(
            std::async(std::launch::async, [&mcts_engine, game_id, &config, &progress]() {
                return play_single_game(mcts_engine, game_id, config, progress, false);
            })
        );
    }
    
    // Progress monitoring thread
    std::atomic<bool> monitor_running{true};
    std::thread progress_thread([&]() {
        auto last_report_time = batch_start_time;
        int last_completed = 0;
        
        while (monitor_running.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - batch_start_time);
            auto interval_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_report_time);
            
            int completed = progress.completed_games.load();
            int interval_completed = completed - last_completed;
            
            if (interval_elapsed.count() > 0) {
                double overall_rate = (double)completed / elapsed.count();
                double interval_rate = (double)interval_completed / interval_elapsed.count();
                
                std::cout << "\n=== LIVE PROGRESS ===" << std::endl;
                std::cout << "Games completed: " << completed << "/" << NUM_GAMES 
                         << " (" << std::fixed << std::setprecision(1) 
                         << 100.0 * completed / NUM_GAMES << "%)" << std::endl;
                std::cout << "Elapsed time: " << elapsed.count() << "s" << std::endl;
                std::cout << "Overall rate: " << std::fixed << std::setprecision(2) 
                         << overall_rate << " games/s" << std::endl;
                std::cout << "Recent rate: " << std::fixed << std::setprecision(2) 
                         << interval_rate << " games/s" << std::endl;
                
                if (overall_rate > 0) {
                    int remaining = NUM_GAMES - completed;
                    std::cout << "Estimated time remaining: " << std::fixed << std::setprecision(1) 
                             << remaining / overall_rate << "s" << std::endl;
                }
                
                int current_total_moves = progress.total_moves_played.load();
                std::cout << "Total moves played (live): " << current_total_moves << std::endl;
                std::cout << "Avg moves per completed game: " << std::fixed << std::setprecision(1) 
                         << (completed > 0 ? (double)current_total_moves / completed : 0.0) << std::endl;
                std::cout << "Estimated total evaluations: ~" << current_total_moves * 800 << std::endl;
                
                // Real-time batch utilization analysis
                if (elapsed.count() > 0) {
                    int estimated_requests_per_sec = (current_total_moves * 800) / elapsed.count();
                    double estimated_batches_per_sec = (double)estimated_requests_per_sec / config.batch_size;
                    std::cout << "Est. eval requests/sec: " << estimated_requests_per_sec << std::endl;
                    std::cout << "Est. batches/sec: " << std::fixed << std::setprecision(1) 
                             << estimated_batches_per_sec << " (batch_size=" << config.batch_size << ")" << std::endl;
                }
                
                std::cout << "Current results: W:" << progress.white_wins.load() 
                         << " B:" << progress.black_wins.load() 
                         << " D:" << progress.draws.load() << std::endl;
                std::cout << "==================\n" << std::endl;
                
                last_report_time = now;
                last_completed = completed;
            }
        }
    });
    
    // Collect results as they complete
    for (int i = 0; i < NUM_GAMES; ++i) {
        GameResult result = game_futures[i].get();
        progress.add_result(result);
        
        // Save individual game PGN
        std::string timestamp = get_timestamp();
        std::string random_suffix = get_random_suffix();
        std::string filename = pgn_folder + "/parallel_game_" + std::to_string(result.game_id) + "_" + 
                              timestamp + "_" + random_suffix + ".pgn";
        
        std::ofstream pgn_file(filename);
        if (pgn_file.is_open()) {
            pgn_file << result.pgn;
            pgn_file.close();
        }
    }
    
    // Stop progress monitoring
    monitor_running.store(false);
    progress_thread.join();
    
    auto batch_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(batch_end_time - batch_start_time);
    
    // Get final statistics from progress tracker
    int white_wins = progress.white_wins.load();
    int black_wins = progress.black_wins.load();
    int draws = progress.draws.load();
    int total_moves = progress.total_moves_played.load();
    
    std::cout << "\n===============================================" << std::endl;
    std::cout << "=== FINAL PARALLEL BATCH STATISTICS ===" << std::endl;
    std::cout << "===============================================" << std::endl;
    std::cout << "Total parallel games played: " << NUM_GAMES << std::endl;
    std::cout << "Total wall-clock time: " << total_duration.count() << " seconds" << std::endl;
    std::cout << "Effective games per second: " << std::fixed << std::setprecision(2) 
              << (double)NUM_GAMES / total_duration.count() << std::endl;
    std::cout << "Parallelization efficiency: " << std::fixed << std::setprecision(1) 
              << 100.0 * NUM_GAMES / (total_duration.count() * (double)NUM_GAMES / total_duration.count()) << "%" << std::endl;
    
    std::cout << "\nGame results breakdown:" << std::endl;
    std::cout << "  White wins: " << white_wins << " (" << std::fixed << std::setprecision(1) 
              << 100.0 * white_wins / NUM_GAMES << "%)" << std::endl;
    std::cout << "  Black wins: " << black_wins << " (" << std::fixed << std::setprecision(1) 
              << 100.0 * black_wins / NUM_GAMES << "%)" << std::endl;
    std::cout << "  Draws: " << draws << " (" << std::fixed << std::setprecision(1) 
              << 100.0 * draws / NUM_GAMES << "%)" << std::endl;
    
    std::cout << "\nGame statistics:" << std::endl;
    std::cout << "  Average moves per game: " << std::fixed << std::setprecision(1) 
              << (double)total_moves / NUM_GAMES << std::endl;
    std::cout << "  Total moves played: " << total_moves << std::endl;
    std::cout << "  Total positions evaluated: ~" << total_moves * 800 << " (estimated)" << std::endl;
    
    // Calculate theoretical batch utilization
    int theoretical_requests_per_second = (total_moves * 800) / total_duration.count();
    std::cout << "\nBatch utilization analysis:" << std::endl;
    std::cout << "  Estimated evaluation requests/sec: " << theoretical_requests_per_second << std::endl;
    std::cout << "  Batch size: " << config.batch_size << std::endl;
    std::cout << "  Theoretical batches/sec: " << std::fixed << std::setprecision(1) 
              << (double)theoretical_requests_per_second / config.batch_size << std::endl;
    
    // Save summary to file
    std::string summary_filename = pgn_folder + "/parallel_summary_" + get_timestamp() + ".txt";
    std::ofstream summary_file(summary_filename);
    if (summary_file.is_open()) {
        summary_file << "MCTS Self-Play Parallel Batch Results\n";
        summary_file << "=====================================\n\n";
        summary_file << "Execution mode: " << NUM_GAMES << " simultaneous parallel games\n";
        summary_file << "Total games: " << NUM_GAMES << "\n";
        summary_file << "Wall-clock time: " << total_duration.count() << " seconds\n";
        summary_file << "Games per second: " << std::fixed << std::setprecision(2) 
                     << (double)NUM_GAMES / total_duration.count() << "\n\n";
        summary_file << "Results:\n";
        summary_file << "White wins: " << white_wins << " (" << std::fixed << std::setprecision(1) 
                     << 100.0 * white_wins / NUM_GAMES << "%)\n";
        summary_file << "Black wins: " << black_wins << " (" << std::fixed << std::setprecision(1) 
                     << 100.0 * black_wins / NUM_GAMES << "%)\n";
        summary_file << "Draws: " << draws << " (" << std::fixed << std::setprecision(1) 
                     << 100.0 * draws / NUM_GAMES << "%)\n\n";
        summary_file << "Statistics:\n";
        summary_file << "Average moves per game: " << std::fixed << std::setprecision(1) 
                     << (double)total_moves / NUM_GAMES << "\n";
        summary_file << "Total moves: " << total_moves << "\n";
        summary_file << "Estimated total evaluations: " << total_moves * 800 << "\n";
        summary_file << "Batch size: " << config.batch_size << "\n";
        summary_file.close();
        std::cout << "Summary saved to: " << summary_filename << std::endl;
    }
    
    std::cout << "===============================================" << std::endl;
    
    return 0;
}