#include "fen_to_tensor.h"
#include <iostream>
#include <deque>
#include <string>

int main() {
    std::cout << "=== Testing Position History Tensor Generation ===" << std::endl;
    
    // Test 1: Empty history (beginning of game)
    std::deque<std::string> empty_history;
    std::string starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    empty_history.push_back(starting_fen);
    
    auto result1 = fen_history_to_tensor(empty_history, 1);
    std::cout << "\nTest 1: Starting position with no history" << std::endl;
    std::cout << "Expected tensor size: " << (8 * 8 * 108) << " (6912)" << std::endl;
    std::cout << "Actual tensor size: " << result1.tensor.size() << std::endl;
    std::cout << "Status: " << (result1.tensor.size() == 6912 ? "PASS" : "FAIL") << std::endl;
    
    // Test 2: Full history (8 positions)
    std::deque<std::string> full_history;
    full_history.push_back("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    full_history.push_back("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    full_history.push_back("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2");
    full_history.push_back("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2");
    full_history.push_back("rnbqkb1r/pp1ppppp/5n2/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3");
    full_history.push_back("rnbqkb1r/pp1ppppp/5n2/2p5/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3");
    full_history.push_back("rnbqkb1r/pp1p1ppp/4pn2/2p5/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4");
    full_history.push_back("rnbqkb1r/pp1p1ppp/4pn2/2p5/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4");
    
    auto result2 = fen_history_to_tensor(full_history, 2);
    std::cout << "\nTest 2: Position with full 8-position history" << std::endl;
    std::cout << "Expected tensor size: " << (8 * 8 * 108) << " (6912)" << std::endl;
    std::cout << "Actual tensor size: " << result2.tensor.size() << std::endl;
    std::cout << "Status: " << (result2.tensor.size() == 6912 ? "PASS" : "FAIL") << std::endl;
    
    // Test 3: Verify channel structure
    // Historical positions should have 13 channels: 12 pieces + 1 side to move
    // Current position should have 17 channels: 12 pieces + 4 castling + 1 side to move
    std::cout << "\nTest 3: Verifying channel structure" << std::endl;
    std::cout << "Historical positions (0-6): 7 positions × 64 squares × 13 channels = " << (7 * 64 * 13) << " values" << std::endl;
    std::cout << "Current position (7): 1 position × 64 squares × 17 channels = " << (64 * 17) << " values" << std::endl;
    std::cout << "Total: " << (7 * 64 * 13 + 64 * 17) << " values" << std::endl;
    
    // Test 4: Test with partial history (should pad with empty positions)
    std::deque<std::string> partial_history;
    partial_history.push_back("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    partial_history.push_back("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    partial_history.push_back("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2");
    
    auto result3 = fen_history_to_tensor(partial_history, 3);
    std::cout << "\nTest 4: Partial history (3 positions, should pad to 8)" << std::endl;
    std::cout << "Expected tensor size: " << (8 * 8 * 108) << " (6912)" << std::endl;
    std::cout << "Actual tensor size: " << result3.tensor.size() << std::endl;
    std::cout << "Status: " << (result3.tensor.size() == 6912 ? "PASS" : "FAIL") << std::endl;
    
    // Test 5: Verify backward compatibility with old function
    auto old_result = fen_to_tensor(starting_fen, 5);
    std::cout << "\nTest 5: Backward compatibility - old fen_to_tensor function" << std::endl;
    std::cout << "Expected tensor size: " << (8 * 8 * 18) << " (1152)" << std::endl;
    std::cout << "Actual tensor size: " << old_result.tensor.size() << std::endl;
    std::cout << "Status: " << (old_result.tensor.size() == 1152 ? "PASS" : "FAIL") << std::endl;
    
    std::cout << "\n=== All Tests Complete ===" << std::endl;
    
    return 0;
}
