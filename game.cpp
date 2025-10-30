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
#include <Python.h>
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
        
        size_t capture_pos = move_str.find(" (capture)");
        if (capture_pos != std::string::npos) {
            move_str = move_str.substr(0, capture_pos);
        }
        
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

struct GameResult {
    std::vector<Move> moves;
    std::string result;
    std::string pgn;
    int move_count;
    std::string starting_fen;
    int game_id;
    double total_time_ms;
    int total_iterations;
};

struct ProgressTracker {
    std::atomic<int> completed_games{0};
    std::atomic<int> total_moves{0};
    std::atomic<int> total_moves_played{0};
    std::atomic<int> white_wins{0};
    std::atomic<int> black_wins{0};
    std::atomic<int> draws{0};
    std::atomic<double> total_move_time_ms{0.0};
    std::atomic<long long> total_iterations{0};
    std::mutex results_mutex;
    std::vector<GameResult> all_results;
    
    void add_result(const GameResult& result) {
        completed_games.fetch_add(1);
        total_moves.fetch_add(result.move_count);
        total_iterations.fetch_add(result.total_iterations);
        
        // Add to total move time
        double current = total_move_time_ms.load();
        while (!total_move_time_ms.compare_exchange_weak(current, current + result.total_time_ms));
        
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

GameResult play_single_game(MCTS& mcts_engine, int game_id, const MCTSConfig& config, ProgressTracker& progress, int iterations_per_move, bool verbose = false) {
    Position pos;
    std::string starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    Position::set(starting_fen, pos);
    
    std::vector<Move> moves_played;
    std::string game_result = "*";
    
    std::map<uint64_t, int> position_history;
    position_history[pos.get_hash()] = 1;
    
    std::deque<std::string> fen_history;
    
    int full_move_count = 0;
    const int MAX_MOVES = 200;
    
    double game_total_time_ms = 0.0;
    int game_total_iterations = 0;
    
    if (verbose) {
        std::cout << "\n=== GAME " << game_id << " ===" << std::endl;
        std::cout << "Starting FEN: " << starting_fen << std::endl;
    }
    
    while (full_move_count < MAX_MOVES) {
        bool is_white = (pos.turn() == WHITE);
        
        if (full_move_count < 20) {
            mcts_engine.set_dirichlet_alpha(0.35);
        } else {
            mcts_engine.set_dirichlet_alpha(0.10);
        }
        
        auto move_start = std::chrono::high_resolution_clock::now();
        Move best_move = mcts_engine.run_search(pos, iterations_per_move, false, fen_history);
        auto move_end = std::chrono::high_resolution_clock::now();
        
        double move_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(move_end - move_start).count();
        game_total_time_ms += move_time_ms;
        game_total_iterations += iterations_per_move;
        
        if (best_move == Move()) {
            if (!is_white) {
                game_result = "0-1";
            } else {
                game_result = "1-0";
            }
            break;
        }
        
        fen_history.push_back(pos.fen());
        if (fen_history.size() > 7) {
            fen_history.pop_front();
        }
        
        moves_played.push_back(best_move);
        
        if (is_white) {
            pos.play<WHITE>(best_move);
        } else {
            pos.play<BLACK>(best_move);
        }
        
        full_move_count++;
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
    result.total_time_ms = game_total_time_ms;
    result.total_iterations = game_total_iterations;
    
    return result;
}

int main() {
    std::cout << "=== Chess Engine Starting ===" << std::endl;
    
    // Initialize Python interpreter
    std::cout << "Initializing Python interpreter..." << std::endl;
    if (!Py_IsInitialized()) {
        Py_Initialize();
        PyEval_InitThreads();
    }
    std::cout << "Python initialized" << std::endl;
    
    // Initialize chess databases
    std::cout << "Initializing chess databases..." << std::endl;
    initialise_all_databases();
    zobrist::initialise_zobrist_keys();
    MoveMappings::initialize_move_mappings();
    std::cout << "Chess databases initialized" << std::endl;
    
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
    config.num_threads = 8;
    config.batch_size = 256;
    config.c_puct = 3.0;
    config.temperature = 1.5;
    config.verbose = false;
    config.dirichlet_alpha = 0.25;
    config.dirichlet_epsilon = 0.2;
    
    std::string model_path = "model";
    int min_batch_size = 16;
    const int NUM_GAMES = 5;
    const int ITERATIONS_PER_MOVE = 800;
    
    std::cout << "\n=== Configuration ===" << std::endl;
    std::cout << "Model path: " << model_path << std::endl;
    std::cout << "Threads: " << config.num_threads << std::endl;
    std::cout << "Max batch size: " << config.batch_size << std::endl;
    std::cout << "Min batch size: " << min_batch_size << std::endl;
    std::cout << "C-PUCT: " << config.c_puct << std::endl;
    std::cout << "Temperature: " << config.temperature << std::endl;
    std::cout << "Number of parallel games: " << NUM_GAMES << std::endl;
    std::cout << "Iterations per move: " << ITERATIONS_PER_MOVE << std::endl;
    
    try {
        // Create evaluator (local model)
        std::cout << "\nInitializing evaluator..." << std::endl;
        auto evaluator = std::make_unique<RemoteEvaluator>(model_path, config.batch_size, min_batch_size);
        std::cout << "Evaluator initialized successfully\n" << std::endl;
        
        // Create MCTS engine
        std::cout << "Creating MCTS engine..." << std::endl;
        MCTS mcts_engine(config, std::move(evaluator));
        std::cout << "MCTS engine created\n" << std::endl;
        
        ProgressTracker progress;
        
        auto batch_start_time = std::chrono::high_resolution_clock::now();
        
        std::cout << "Starting " << NUM_GAMES << " parallel games..." << std::endl;
        
        // Launch all games in parallel
        std::vector<std::future<GameResult>> game_futures;
        game_futures.reserve(NUM_GAMES);
        
        for (int game_id = 1; game_id <= NUM_GAMES; ++game_id) {
            game_futures.emplace_back(
                std::async(std::launch::async, [&mcts_engine, game_id, &config, &progress, ITERATIONS_PER_MOVE]() {
                    return play_single_game(mcts_engine, game_id, config, progress, ITERATIONS_PER_MOVE, false);
                })
            );
        }
        
        // Progress monitoring thread
        std::atomic<bool> monitor_running{true};
        std::thread progress_thread([&]() {
            auto last_report_time = batch_start_time;
            int last_completed = 0;
            int last_moves = 0;
            long long last_iterations = 0;
            
            while (monitor_running.load()) {
                std::this_thread::sleep_for(std::chrono::seconds(10));
                
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - batch_start_time);
                auto interval_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_report_time);
                
                int completed = progress.completed_games.load();
                int interval_completed = completed - last_completed;
                int current_moves = progress.total_moves_played.load();
                int interval_moves = current_moves - last_moves;
                long long current_iterations = progress.total_iterations.load();
                long long interval_iterations = current_iterations - last_iterations;
                
                if (interval_elapsed.count() > 0) {
                    double overall_rate = (double)completed / elapsed.count();
                    double interval_rate = (double)interval_completed / interval_elapsed.count();
                    double moves_per_sec = (double)interval_moves / interval_elapsed.count();
                    double iterations_per_sec = (double)interval_iterations / interval_elapsed.count();
                    
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
                    
                    std::cout << "\nMove Statistics:" << std::endl;
                    std::cout << "Total moves played: " << current_moves << std::endl;
                    std::cout << "Moves in last interval: " << interval_moves << std::endl;
                    std::cout << "Moves per second: " << std::fixed << std::setprecision(2) 
                             << moves_per_sec << std::endl;
                    
                    if (completed > 0) {
                        double avg_time_per_move = progress.total_move_time_ms.load() / current_moves;
                        std::cout << "Average time per move: " << std::fixed << std::setprecision(2) 
                                 << avg_time_per_move << " ms" << std::endl;
                    }
                    
                    std::cout << "Avg moves per completed game: " << std::fixed << std::setprecision(1) 
                             << (completed > 0 ? (double)current_moves / completed : 0.0) << std::endl;
                    
                    std::cout << "\nSearch Statistics:" << std::endl;
                    std::cout << "Total iterations: " << current_iterations << std::endl;
                    std::cout << "Iterations in last interval: " << interval_iterations << std::endl;
                    std::cout << "Search speed: " << std::fixed << std::setprecision(2) 
                             << iterations_per_sec << " iterations/second" << std::endl;
                    
                    if (interval_moves > 0) {
                        double avg_iterations_per_move = (double)interval_iterations / interval_moves;
                        std::cout << "Avg iterations per move: " << std::fixed << std::setprecision(0) 
                                 << avg_iterations_per_move << std::endl;
                    }
                    
                    std::cout << "\nCurrent results: W:" << progress.white_wins.load() 
                             << " B:" << progress.black_wins.load() 
                             << " D:" << progress.draws.load() << std::endl;
                    std::cout << "==================\n" << std::endl;
                    
                    last_report_time = now;
                    last_completed = completed;
                    last_moves = current_moves;
                    last_iterations = current_iterations;
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
        
        int white_wins = progress.white_wins.load();
        int black_wins = progress.black_wins.load();
        int draws = progress.draws.load();
        int total_moves = progress.total_moves_played.load();
        double total_time_ms = progress.total_move_time_ms.load();
        long long total_iterations = progress.total_iterations.load();
        
        std::cout << "\n===============================================" << std::endl;
        std::cout << "=== FINAL STATISTICS ===" << std::endl;
        std::cout << "===============================================" << std::endl;
        std::cout << "Total games played: " << NUM_GAMES << std::endl;
        std::cout << "Total wall-clock time: " << total_duration.count() << " seconds" << std::endl;
        std::cout << "Games per second: " << std::fixed << std::setprecision(2) 
                  << (double)NUM_GAMES / total_duration.count() << std::endl;
        
        std::cout << "\nGame results:" << std::endl;
        std::cout << "  White wins: " << white_wins << " (" << std::fixed << std::setprecision(1) 
                  << 100.0 * white_wins / NUM_GAMES << "%)" << std::endl;
        std::cout << "  Black wins: " << black_wins << " (" << std::fixed << std::setprecision(1) 
                  << 100.0 * black_wins / NUM_GAMES << "%)" << std::endl;
        std::cout << "  Draws: " << draws << " (" << std::fixed << std::setprecision(1) 
                  << 100.0 * draws / NUM_GAMES << "%)" << std::endl;
        
        std::cout << "\nMove statistics:" << std::endl;
        std::cout << "  Total moves: " << total_moves << std::endl;
        std::cout << "  Average moves per game: " << std::fixed << std::setprecision(1) 
                  << (double)total_moves / NUM_GAMES << std::endl;
        std::cout << "  Average time per move: " << std::fixed << std::setprecision(2) 
                  << total_time_ms / total_moves << " ms" << std::endl;
        std::cout << "  Moves per second: " << std::fixed << std::setprecision(2) 
                  << (total_moves * 1000.0) / total_time_ms << std::endl;
        
        std::cout << "\nSearch statistics:" << std::endl;
        std::cout << "  Total iterations: " << total_iterations << std::endl;
        std::cout << "  Iterations per move: " << ITERATIONS_PER_MOVE << std::endl;
        std::cout << "  Average search speed: " << std::fixed << std::setprecision(2) 
                  << (total_iterations * 1000.0) / total_time_ms << " iterations/second" << std::endl;
        std::cout << "  Time per iteration: " << std::fixed << std::setprecision(3) 
                  << total_time_ms / total_iterations << " ms" << std::endl;
        
        std::cout << "===============================================" << std::endl;
        
        // Allow cleanup
        std::cout << "\nWaiting for cleanup..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
    } catch (const std::exception& e) {
        std::cerr << "\n!!! ERROR !!!" << std::endl;
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "=== Chess Engine Shutdown Complete ===" << std::endl;
    
    return 0;
}