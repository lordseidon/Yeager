#include "position_tensor.h"
#include "types.h"
#include <vector>
#include <array>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <algorithm>

// Create move mappings similar to Python version
namespace MoveMappings {
    std::unordered_map<std::string, int> move_to_index;
    std::unordered_map<int, std::string> index_to_move;
    int policy_output_size = 0;
    bool initialized = false;

    void initialize_move_mappings() {
        if (initialized) return;
        
        std::vector<std::string> moves;
        
        // Generate all possible moves (from_square, to_square combinations)
        for (int from_sq = 0; from_sq < 64; from_sq++) {
            for (int to_sq = 0; to_sq < 64; to_sq++) {
                int from_rank = from_sq / 8;
                int to_rank = to_sq / 8;
                
                // Check for promotion moves
                bool is_white_promo = (from_rank == 6 && to_rank == 7);
                bool is_black_promo = (from_rank == 1 && to_rank == 0);
                
                if (is_white_promo || is_black_promo) {
                    // Add promotion moves for each piece type
                    char promotions[] = {'q', 'r', 'b', 'n'};
                    for (char promo : promotions) {
                        std::string move_uci = square_to_string(static_cast<Square>(from_sq)) + 
                                              square_to_string(static_cast<Square>(to_sq)) + promo;
                        moves.push_back(move_uci);
                    }
                } else {
                    std::string move_uci = square_to_string(static_cast<Square>(from_sq)) + 
                                          square_to_string(static_cast<Square>(to_sq));
                    moves.push_back(move_uci);
                }
            }
        }
        
        // Remove duplicates and sort
        std::sort(moves.begin(), moves.end());
        moves.erase(std::unique(moves.begin(), moves.end()), moves.end());
        
        // Create mappings
        for (size_t i = 0; i < moves.size(); i++) {
            move_to_index[moves[i]] = static_cast<int>(i);
            index_to_move[static_cast<int>(i)] = moves[i];
        }
        
        policy_output_size = static_cast<int>(moves.size());
        initialized = true;
        
        std::cout << "Generated " << policy_output_size << " unique moves for the policy map." << std::endl;
    }
    
    int move_to_policy_index(Move move) {
        if (!initialized) initialize_move_mappings();
        
        std::string move_str = move_to_string(move);
        auto it = move_to_index.find(move_str);
        return (it != move_to_index.end()) ? it->second : -1;
    }
}

// Helper function to convert square to string (e.g., A1, B2, etc.)
std::string square_to_string(Square sq) {
    if (sq == NO_SQUARE) return "none";
    
    // Use your engine's SQSTR array if available, otherwise compute manually
    if (sq >= 0 && sq < 64) {
        // Manual conversion to be safe
        int file = sq % 8;  // 0-7 for files a-h
        int rank = sq / 8;  // 0-7 for ranks 1-8
        
        char file_char = 'a' + file;
        char rank_char = '1' + rank;
        
        return std::string(1, file_char) + std::string(1, rank_char);
    }
    
    return "invalid";
}

// Helper function to convert move to UCI string
std::string move_to_string(Move move) {
    // Check if it's a null/empty move - adjust this based on your Move class
    // You might need to check for a specific value or use a different method
    
    Square from = move.from();
    Square to = move.to();
    
    std::string result = square_to_string(from) + square_to_string(to);
    
    // Handle promotion
    MoveFlags flags = move.flags();
    if (flags == PR_QUEEN || flags == PC_QUEEN) {
        result += 'q';
    } else if (flags == PR_ROOK || flags == PC_ROOK) {
        result += 'r';
    } else if (flags == PR_BISHOP || flags == PC_BISHOP) {
        result += 'b';
    } else if (flags == PR_KNIGHT || flags == PC_KNIGHT) {
        result += 'n';
    }
    
    return result;
}

// Convert position to tensor (8x8x18 format, channels last)
std::vector<float> position_to_tensor(const Position& pos) {
    // Initialize tensor: 8x8x18 = 1152 elements
    std::vector<float> tensor(8 * 8 * 18, 0.0f);
    
    Color player_color = pos.turn();
    
    // Fill piece planes (channels 0-11)
    for (int sq = 0; sq < 64; ++sq) {
        Square square = static_cast<Square>(sq);
        Piece piece = pos.at(square);
        if (piece != NO_PIECE) {
            int rank = sq / 8;  // 0-7
            int file = sq % 8;  // 0-7
            
            // Flip board perspective for black
            if (player_color == BLACK) {
                rank = 7 - rank;
            }
            
            PieceType pt = type_of(piece);
            Color pc = color_of(piece);
            
            int piece_type_idx = static_cast<int>(pt) - 1; // PAWN=1 becomes 0, etc.
            int channel;
            
            if (pc == player_color) {
                channel = piece_type_idx;  // Own pieces: channels 0-5
            } else {
                channel = piece_type_idx + 6;  // Opponent pieces: channels 6-11
            }
            
            int tensor_idx = rank * 8 * 18 + file * 18 + channel;
            tensor[tensor_idx] = 1.0f;
        }
    }
    
    // Castling rights planes (channels 12-15)
    // Check if castling squares haven't been moved from based on UndoInfo entry bitboard
    bool white_oo = !(pos.history[pos.ply()].entry & (SQUARE_BB[e1] | SQUARE_BB[h1]));
    bool white_ooo = !(pos.history[pos.ply()].entry & (SQUARE_BB[e1] | SQUARE_BB[a1]));
    bool black_oo = !(pos.history[pos.ply()].entry & (SQUARE_BB[e8] | SQUARE_BB[h8]));
    bool black_ooo = !(pos.history[pos.ply()].entry & (SQUARE_BB[e8] | SQUARE_BB[a8]));
    
    if (player_color == WHITE) {
        if (white_oo) {
            // Fill entire plane for white kingside castling
            for (int i = 0; i < 8 * 8; i++) {
                tensor[i * 18 + 12] = 1.0f;
            }
        }
        if (white_ooo) {
            // Fill entire plane for white queenside castling
            for (int i = 0; i < 8 * 8; i++) {
                tensor[i * 18 + 13] = 1.0f;
            }
        }
        if (black_oo) {
            // Fill entire plane for black kingside castling
            for (int i = 0; i < 8 * 8; i++) {
                tensor[i * 18 + 14] = 1.0f;
            }
        }
        if (black_ooo) {
            // Fill entire plane for black queenside castling
            for (int i = 0; i < 8 * 8; i++) {
                tensor[i * 18 + 15] = 1.0f;
            }
        }
    } else {
        // From black's perspective
        if (black_oo) {
            for (int i = 0; i < 8 * 8; i++) {
                tensor[i * 18 + 12] = 1.0f;
            }
        }
        if (black_ooo) {
            for (int i = 0; i < 8 * 8; i++) {
                tensor[i * 18 + 13] = 1.0f;
            }
        }
        if (white_oo) {
            for (int i = 0; i < 8 * 8; i++) {
                tensor[i * 18 + 14] = 1.0f;
            }
        }
        if (white_ooo) {
            for (int i = 0; i < 8 * 8; i++) {
                tensor[i * 18 + 15] = 1.0f;
            }
        }
    }
    
    // En passant plane (channel 16)
    if (pos.history[pos.ply()].epsq != NO_SQUARE) {
        Square ep_sq = pos.history[pos.ply()].epsq;
        int rank = ep_sq / 8;
        int file = ep_sq % 8;
        
        // Flip board perspective for black
        if (player_color == BLACK) {
            rank = 7 - rank;
        }
        
        int tensor_idx = rank * 8 * 18 + file * 18 + 16;
        tensor[tensor_idx] = 1.0f;
    }
    
    // Player color plane (channel 17)
    float color_value = (player_color == WHITE) ? 1.0f : 0.0f;
    for (int i = 0; i < 8 * 8; i++) {
        tensor[i * 18 + 17] = color_value;
    }
    
    return tensor;
}

// Convenience function to get tensor as 3D array access
float get_tensor_value(const std::vector<float>& tensor, int rank, int file, int channel) {
    if (rank < 0 || rank >= 8 || file < 0 || file >= 8 || channel < 0 || channel >= 18) {
        return 0.0f; // Out of bounds
    }
    return tensor[rank * 8 * 18 + file * 18 + channel];
}

// Set tensor value in 3D array
void set_tensor_value(std::vector<float>& tensor, int rank, int file, int channel, float value) {
    if (rank < 0 || rank >= 8 || file < 0 || file >= 8 || channel < 0 || channel >= 18) {
        return; // Out of bounds
    }
    tensor[rank * 8 * 18 + file * 18 + channel] = value;
}

// Print tensor for debugging
void print_tensor_summary(const std::vector<float>& tensor) {
    std::cout << "Tensor summary (8x8x18):" << std::endl;
    
    // Print non-zero channels
    for (int channel = 0; channel < 18; channel++) {
        bool has_values = false;
        int count = 0;
        
        for (int rank = 0; rank < 8; rank++) {
            for (int file = 0; file < 8; file++) {
                if (get_tensor_value(tensor, rank, file, channel) != 0.0f) {
                    has_values = true;
                    count++;
                }
            }
        }
        
        if (has_values) {
            std::string channel_name;
            if (channel < 6) {
                const char* pieces[] = {"Pawn", "Knight", "Bishop", "Rook", "Queen", "King"};
                channel_name = std::string("Own ") + pieces[channel];
            } else if (channel < 12) {
                const char* pieces[] = {"Pawn", "Knight", "Bishop", "Rook", "Queen", "King"};
                channel_name = std::string("Opponent ") + pieces[channel - 6];
            } else if (channel == 12) {
                channel_name = "Own Kingside Castling";
            } else if (channel == 13) {
                channel_name = "Own Queenside Castling";
            } else if (channel == 14) {
                channel_name = "Opponent Kingside Castling";
            } else if (channel == 15) {
                channel_name = "Opponent Queenside Castling";
            } else if (channel == 16) {
                channel_name = "En Passant";
            } else if (channel == 17) {
                channel_name = "Player Color";
            }
            
            std::cout << "  Channel " << channel << " (" << channel_name << "): " 
                      << count << " non-zero values" << std::endl;
        }
    }
}

// Print detailed tensor values for specific channels
void print_tensor_channel(const std::vector<float>& tensor, int channel) {
    std::cout << "\nChannel " << channel << " values:" << std::endl;
    std::cout << "   a  b  c  d  e  f  g  h" << std::endl;
    
    for (int rank = 7; rank >= 0; rank--) {  // Print from rank 8 down to rank 1
        std::cout << (rank + 1) << " ";
        for (int file = 0; file < 8; file++) {
            float value = get_tensor_value(tensor, rank, file, channel);
            std::cout << " " << (value > 0.5f ? "1" : ".") << " ";
        }
        std::cout << std::endl;
    }
}

// Print all piece channels in a readable format
void print_tensor_detailed(const std::vector<float>& tensor) {
    std::cout << "\nDetailed Tensor Analysis:" << std::endl;
    
    // Print piece channels
    const char* piece_names[] = {"Pawn", "Knight", "Bishop", "Rook", "Queen", "King"};
    
    for (int piece_type = 0; piece_type < 6; piece_type++) {
        // Own pieces
        int own_channel = piece_type;
        int opp_channel = piece_type + 6;
        
        bool own_has_pieces = false;
        bool opp_has_pieces = false;
        
        for (int sq = 0; sq < 64; sq++) {
            int rank = sq / 8, file = sq % 8;
            if (get_tensor_value(tensor, rank, file, own_channel) > 0.5f) own_has_pieces = true;
            if (get_tensor_value(tensor, rank, file, opp_channel) > 0.5f) opp_has_pieces = true;
        }
        
        if (own_has_pieces) {
            std::cout << "\nOwn " << piece_names[piece_type] << "s (Channel " << own_channel << "):" << std::endl;
            print_tensor_channel(tensor, own_channel);
        }
        
        if (opp_has_pieces) {
            std::cout << "\nOpponent " << piece_names[piece_type] << "s (Channel " << opp_channel << "):" << std::endl;
            print_tensor_channel(tensor, opp_channel);
        }
    }
    
    // Print color channel
    std::cout << "\nPlayer Color (Channel 17):" << std::endl;
    float color_value = get_tensor_value(tensor, 0, 0, 17);
    std::cout << "Value: " << color_value << " (1.0 = WHITE, 0.0 = BLACK)" << std::endl;
}