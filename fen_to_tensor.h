#ifndef FEN_TO_TENSOR_H
#define FEN_TO_TENSOR_H

#include <string>
#include <deque>
#include <iostream>
#include <sstream>
#include <cctype>
#include <iomanip> // For printing utilities

// A struct to hold both the ID and the resulting tensor
template <typename IDType>
struct TensorResult {
    IDType id;
    std::deque
<float> tensor;
};

// The fen_to_tensor function is now a template and its full implementation resides in the header.
// It takes a FEN string and an ID of any type, and returns a TensorResult struct.
template <typename IDType>
TensorResult<IDType> fen_to_tensor(const std::string& fen, const IDType& id) {
    // Initialize tensor: 8x8x18 = 1152 elements
    std::deque
<float> tensor(8 * 8 * 18, 0.0f);
    
    // Parse FEN string
    std::istringstream fen_stream(fen);
    std::string board_str, turn_str, castling_str, ep_str, halfmove_str, fullmove_str;
    
    fen_stream >> board_str >> turn_str >> castling_str >> ep_str >> halfmove_str >> fullmove_str;
    
    // Determine player color
    bool is_white_turn = (turn_str == "w");
    
    // --- BOARD POSITION PARSING ---
    int rank = 7;
    int file = 0;
    
    for (char c : board_str) {
        if (c == '/') {
            rank--;
            file = 0;
        } else if (std::isdigit(c)) {
            file += (c - '0');  
        } else {
            bool piece_is_white = std::isupper(c);
            char piece_char = std::tolower(c);
            
            int piece_type_idx = -1;
            switch (piece_char) {
                case 'p': piece_type_idx = 0; break;
                case 'n': piece_type_idx = 1; break;
                case 'b': piece_type_idx = 2; break;
                case 'r': piece_type_idx = 3; break;
                case 'q': piece_type_idx = 4; break;
                case 'k': piece_type_idx = 5; break;
            }
            
            if (piece_type_idx != -1) {
                int tensor_rank = rank;
                int tensor_file = file;
                
                if (!is_white_turn) {
                    tensor_rank = 7 - rank;
                    tensor_file = 7 - file;
                }
                
                int channel = (piece_is_white == is_white_turn) ? piece_type_idx : piece_type_idx + 6;
                int tensor_idx = (tensor_rank * 8 + tensor_file) * 18 + channel;
                tensor[tensor_idx] = 1.0f;
            }
            file++;
        }
    }
    
    // --- METADATA PLANES ---
    bool white_kingside = castling_str.find('K') != std::string::npos;
    bool white_queenside = castling_str.find('Q') != std::string::npos;
    bool black_kingside = castling_str.find('k') != std::string::npos;
    bool black_queenside = castling_str.find('q') != std::string::npos;
    
    if (is_white_turn) {
        if (white_kingside)  for (int i = 0; i < 64; ++i) tensor[i * 18 + 12] = 1.0f;
        if (white_queenside) for (int i = 0; i < 64; ++i) tensor[i * 18 + 13] = 1.0f;
        if (black_kingside)  for (int i = 0; i < 64; ++i) tensor[i * 18 + 14] = 1.0f;
        if (black_queenside) for (int i = 0; i < 64; ++i) tensor[i * 18 + 15] = 1.0f;
    } else {
        if (black_kingside)  for (int i = 0; i < 64; ++i) tensor[i * 18 + 12] = 1.0f;
        if (black_queenside) for (int i = 0; i < 64; ++i) tensor[i * 18 + 13] = 1.0f;
        if (white_kingside)  for (int i = 0; i < 64; ++i) tensor[i * 18 + 14] = 1.0f;
        if (white_queenside) for (int i = 0; i < 64; ++i) tensor[i * 18 + 15] = 1.0f;
    }
    
    if (ep_str != "-") {
        int ep_file = ep_str[0] - 'a';
        int ep_rank = ep_str[1] - '1';
        int tensor_rank = is_white_turn ? ep_rank : 7 - ep_rank;
        int tensor_file = is_white_turn ? ep_file : 7 - ep_file;
        int tensor_idx = (tensor_rank * 8 + tensor_file) * 18 + 16;
        tensor[tensor_idx] = 1.0f;
    }
    
    float color_value = is_white_turn ? 1.0f : 0.0f;
    for (int i = 0; i < 64; ++i) {
        tensor[i * 18 + 17] = color_value;
    }
    
    // Return the struct containing both the ID and the tensor
    return {id, tensor};
}

// --- UTILITY FUNCTION DECLARATIONS ---
float get_tensor_value(const std::deque
<float>& tensor, int rank, int file, int channel);
void print_tensor_summary(const std::deque
<float>& tensor);
void print_tensor_channel(const std::deque
<float>& tensor, int channel);
void print_tensor_detailed(const std::deque
<float>& tensor);
void print_full_tensor_structured(const std::deque
<float>& tensor);
void print_raw_tensor(const std::deque
<float>& tensor);

#endif // FEN_TO_TENSOR_H