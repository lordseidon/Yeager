#include "position_tensor.h"
#include "types.h"
#include <iostream>

int main() {
    std::cout << "=== Testing Move Mappings ===" << std::endl;
    
    // Initialize the mappings
    MoveMappings::initialize_move_mappings();
    
    std::cout << "\nTotal policy output size: " << MoveMappings::policy_output_size << std::endl;
    
    // Test a few specific moves
    std::cout << "\n=== Testing Specific Moves ===" << std::endl;
    
    // Normal move
    std::cout << "Move 'e2e4' -> index: " << MoveMappings::move_to_index["e2e4"] << std::endl;
    std::cout << "Move 'e7e5' -> index: " << MoveMappings::move_to_index["e7e5"] << std::endl;
    
    // White promotion moves (from rank 7 to rank 8, i.e., from_rank=6 to to_rank=7)
    std::cout << "\n=== White Promotions (rank 7->8) ===" << std::endl;
    std::cout << "Move 'e7e8' (normal) -> index: " << MoveMappings::move_to_index["e7e8"] << std::endl;
    std::cout << "Move 'e7e8q' (queen) -> index: " << MoveMappings::move_to_index["e7e8q"] << std::endl;
    std::cout << "Move 'e7e8r' (rook) -> index: " << MoveMappings::move_to_index["e7e8r"] << std::endl;
    std::cout << "Move 'e7e8b' (bishop) -> index: " << MoveMappings::move_to_index["e7e8b"] << std::endl;
    std::cout << "Move 'e7e8n' (knight) -> index: " << MoveMappings::move_to_index["e7e8n"] << std::endl;
    
    // Black promotion moves (from rank 2 to rank 1, i.e., from_rank=1 to to_rank=0)
    std::cout << "\n=== Black Promotions (rank 2->1) ===" << std::endl;
    std::cout << "Move 'e2e1' (normal) -> index: " << MoveMappings::move_to_index["e2e1"] << std::endl;
    std::cout << "Move 'e2e1q' (queen) -> index: " << MoveMappings::move_to_index["e2e1q"] << std::endl;
    std::cout << "Move 'e2e1r' (rook) -> index: " << MoveMappings::move_to_index["e2e1r"] << std::endl;
    std::cout << "Move 'e2e1b' (bishop) -> index: " << MoveMappings::move_to_index["e2e1b"] << std::endl;
    std::cout << "Move 'e2e1n' (knight) -> index: " << MoveMappings::move_to_index["e2e1n"] << std::endl;
    
    // Non-promotion moves that shouldn't have promotion variants
    std::cout << "\n=== Non-Promotion Moves ===" << std::endl;
    std::cout << "Move 'e3e4' -> index: " << MoveMappings::move_to_index["e3e4"] << std::endl;
    std::cout << "Move 'e3e4q' (should not exist) -> index: " << MoveMappings::move_to_index["e3e4q"] << std::endl;
    
    // Show first 20 moves
    std::cout << "\n=== First 20 Moves in Index ===" << std::endl;
    for (int i = 0; i < 20 && i < MoveMappings::policy_output_size; i++) {
        std::cout << "Index " << i << ": " << MoveMappings::index_to_move[i] << std::endl;
    }
    
    return 0;
}
