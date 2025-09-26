#ifndef POSITION_TENSOR_H
#define POSITION_TENSOR_H

#include "types.h"
#include <vector>
#include <string>
#include <unordered_map>
#include "position.h"

// Move mappings namespace for converting moves to policy indices
namespace MoveMappings {
    extern std::unordered_map<std::string, int> move_to_index;
    extern std::unordered_map<int, std::string> index_to_move;
    extern int policy_output_size;
    extern bool initialized;
    
    // Initialize the move mappings (call once at startup)
    void initialize_move_mappings();
    
    // Convert a move to policy network index
    int move_to_policy_index(Move move);
}

// Helper functions for move/square string conversion
std::string square_to_string(Square sq);
std::string move_to_string(Move move);

// Main function: convert position to neural network tensor format
// Returns a flat vector representing an 8x8x18 tensor (channels last)
// Tensor layout:
//   Channels 0-5:   Own pieces (Pawn, Knight, Bishop, Rook, Queen, King)
//   Channels 6-11:  Opponent pieces (Pawn, Knight, Bishop, Rook, Queen, King)
//   Channels 12-15: Castling rights (Own King/Queen side, Opponent King/Queen side)
//   Channel 16:     En passant target square
//   Channel 17:     Player color (1.0 for WHITE, 0.0 for BLACK)
std::vector<float> position_to_tensor(const Position& pos);

// Utility functions for 3D tensor access
float get_tensor_value(const std::vector<float>& tensor, int rank, int file, int channel);
void set_tensor_value(std::vector<float>& tensor, int rank, int file, int channel, float value);

// Debug functions to print tensor information
void print_tensor_summary(const std::vector<float>& tensor);
void print_tensor_channel(const std::vector<float>& tensor, int channel);
void print_tensor_detailed(const std::vector<float>& tensor);

// Constants
constexpr int TENSOR_HEIGHT = 8;
constexpr int TENSOR_WIDTH = 8;
constexpr int TENSOR_CHANNELS = 18;
constexpr int TENSOR_SIZE = TENSOR_HEIGHT * TENSOR_WIDTH * TENSOR_CHANNELS; // 1152

#endif // POSITION_TENSOR_H