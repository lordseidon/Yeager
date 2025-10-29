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
    std::deque<float> tensor;
};

// Helper function to parse a single position's board state (12 channels for pieces + side to move)
// Used for historical positions (no castling/ep info)
inline std::deque<float> fen_to_board_tensor(const std::string& fen, bool from_perspective_white) {
    std::deque<float> tensor(8 * 8 * 13, 0.0f);
    
    if (fen.empty()) {
        // Should not happen with new logic, but handle gracefully
        return tensor;
    }
    
    std::istringstream fen_stream(fen);
    std::string board_str, turn_str;
    fen_stream >> board_str >> turn_str;
    
    bool is_white_turn = (turn_str == "w");
    
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
                
                if (!from_perspective_white) {
                    tensor_rank = 7 - rank;
                }
                
                int channel = (piece_is_white == from_perspective_white) ? piece_type_idx : piece_type_idx + 6;
                int tensor_idx = (tensor_rank * 8 + tensor_file) * 13 + channel;
                tensor[tensor_idx] = 1.0f;
            }
            file++;
        }
    }
    
    // Side to move channel (channel 12)
    float color_value = is_white_turn ? 1.0f : 0.0f;
    for (int i = 0; i < 64; ++i) {
        tensor[i * 13 + 12] = color_value;
    }
    
    return tensor;
}

// Helper function to flip the side to move in a FEN string
inline std::string flip_side_to_move(const std::string& fen) {
    if (fen.empty()) return "";
    
    std::istringstream fen_stream(fen);
    std::string board_str, turn_str, castling_str, ep_str, halfmove_str, fullmove_str;
    fen_stream >> board_str >> turn_str >> castling_str >> ep_str >> halfmove_str >> fullmove_str;
    
    // Flip the turn
    turn_str = (turn_str == "w") ? "b" : "w";
    
    // Reconstruct FEN
    std::ostringstream result;
    result << board_str << " " << turn_str << " " << castling_str << " " 
           << ep_str << " " << halfmove_str << " " << fullmove_str;
    return result.str();
}

// New function: Takes 8 FEN strings (7 history + 1 current) and produces combined tensor
// If history has fewer than 7 positions, repeat current position with alternating colors
// History positions (0-6): 13 channels each (12 pieces + side to move)
// Current position (7): 17 channels (12 pieces + 4 castling + side to move)
// Total: 7*13 + 17 = 108 channels

template <typename IDType>
TensorResult<IDType> fen_history_to_tensor(const std::deque<std::string>& fen_history, const IDType& id) {
    // === PREPARE HISTORY ===
    std::deque<std::string> fens;
    std::string current_fen = fen_history.empty()
        ? "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        : fen_history.back();

    // === DETERMINE SIDE TO MOVE ===
    std::istringstream cur_stream(current_fen);
    std::string dummy, turn_str;
    cur_stream >> dummy >> turn_str;
    bool from_white_perspective = (turn_str == "w");

    // === BRANCH BASED ON TURN ===
    if (!from_white_perspective) {
        // ===============================================
        // ============ BLACK VERSION (TOP) ===============
        // ===============================================

        if (fen_history.size() >= 8) {
            for (size_t i = fen_history.size() - 8; i < fen_history.size() - 1; ++i)
                fens.push_back(fen_history[i]);
        } else if (fen_history.size() > 1) {
            int actual_history = fen_history.size() - 1;
            int to_fill = 7 - actual_history;
            std::string fill = flip_side_to_move(current_fen);
            for (int i = 0; i < to_fill; ++i) {
                fens.push_back(fill);
                fill = flip_side_to_move(fill);
            }
            for (int i = 0; i < actual_history; ++i)
                fens.push_back(fen_history[i]);
        } else {
            std::string fill = current_fen;
            for (int i = 0; i < 7; ++i) {
                fill = flip_side_to_move(fill);
                fens.push_front(fill);
            }
        }

        if (fens.size() > 7)
            fens.erase(fens.begin(), fens.begin() + (fens.size() - 7));

        std::deque<float> tensor(8 * 8 * 108, 0.0f);

        // === PROCESS HISTORICAL POSITIONS (0-6) ===
        for (int h = 0; h < 7; ++h) {
            std::istringstream fen_stream(fens[h]);
            std::string board_str, turn, castling, ep, halfmove, fullmove;
            fen_stream >> board_str >> turn >> castling >> ep >> halfmove >> fullmove;

            bool is_white_turn = (turn == "w");
            int base = h * 13;

            int rank = 7, file = 0;
            for (char c : board_str) {
                if (c == '/') {
                    rank--;
                    file = 0;
                } else if (isdigit(c)) {
                    file += (c - '0');
                } else {
                    bool piece_is_white = isupper(c);
                    char p = tolower(c);
                    int type = -1;
                    switch (p) {
                        case 'p': type = 0; break;
                        case 'n': type = 1; break;
                        case 'b': type = 2; break;
                        case 'r': type = 3; break;
                        case 'q': type = 4; break;
                        case 'k': type = 5; break;
                    }
                    if (type != -1) {
                        int tr = from_white_perspective ? rank : 7 - rank;
                        int tf = from_white_perspective ? file : file;
                        int channel = (piece_is_white == from_white_perspective) ? type : type + 6;
                        int idx = (tr * 8 + tf) * 108 + base + channel;
                        tensor[idx] = 1.0f;
                    }
                    file++;
                }
            }

            float color_val = (!from_white_perspective) ? 1.0f : 0.0f;
            for (int i = 0; i < 64; ++i)
                tensor[i * 108 + base + 12] = color_val;
        }

        // === CURRENT POSITION (7): 17 channels ===
        std::istringstream cs(current_fen);
        std::string board_str, turn, castling, ep, half, full;
        cs >> board_str >> turn >> castling >> ep >> half >> full;
        bool is_white_turn = (turn == "w");
        int base = 7 * 13;
        int rank = 7, file = 0;

        for (char c : board_str) {
            if (c == '/') {
                rank--;
                file = 0;
            } else if (isdigit(c)) {
                file += (c - '0');
            } else {
                bool piece_is_white = isupper(c);
                char p = tolower(c);
                int type = -1;
                switch (p) {
                    case 'p': type = 0; break;
                    case 'n': type = 1; break;
                    case 'b': type = 2; break;
                    case 'r': type = 3; break;
                    case 'q': type = 4; break;
                    case 'k': type = 5; break;
                }
                if (type != -1) {
                    int tr = from_white_perspective ? rank : 7 - rank;
                    int tf = from_white_perspective ? file : file;
                    int channel = (piece_is_white == from_white_perspective) ? type : type + 6;
                    int idx = (tr * 8 + tf) * 108 + base + channel;
                    tensor[idx] = 1.0f;
                }
                file++;
            }
        }

        float color_val = (from_white_perspective) ? 1.0f : 0.0f;
        for (int i = 0; i < 64; ++i)
            tensor[i * 108 + base + 12] = color_val;

        bool wk = castling.find('K') != std::string::npos;
        bool wq = castling.find('Q') != std::string::npos;
        bool bk = castling.find('k') != std::string::npos;
        bool bq = castling.find('q') != std::string::npos;

        if (is_white_turn) {
            if (wk) for (int i = 0; i < 64; ++i) tensor[i * 108 + base + 13] = 1.0f;
            if (wq) for (int i = 0; i < 64; ++i) tensor[i * 108 + base + 14] = 1.0f;
            if (bk) for (int i = 0; i < 64; ++i) tensor[i * 108 + base + 15] = 1.0f;
            if (bq) for (int i = 0; i < 64; ++i) tensor[i * 108 + base + 16] = 1.0f;
        } else {
            if (bk) for (int i = 0; i < 64; ++i) tensor[i * 108 + base + 13] = 1.0f;
            if (bq) for (int i = 0; i < 64; ++i) tensor[i * 108 + base + 14] = 1.0f;
            if (wk) for (int i = 0; i < 64; ++i) tensor[i * 108 + base + 15] = 1.0f;
            if (wq) for (int i = 0; i < 64; ++i) tensor[i * 108 + base + 16] = 1.0f;
        }

        return {id, tensor};

    } else {
        // ===============================================
        // ============ WHITE VERSION (BOTTOM) ============
        // ===============================================

        if (fen_history.size() >= 8) {
            for (size_t i = fen_history.size() - 8; i < fen_history.size() - 1; ++i)
                fens.push_back(fen_history[i]);
        } else if (fen_history.size() > 1) {
            int actual_history = fen_history.size() - 1;
            int to_fill = 7 - actual_history;
            std::string fill = flip_side_to_move(current_fen);
            for (int i = 0; i < to_fill; ++i) {
                fens.push_back(fill);
                fill = flip_side_to_move(fill);
            }
            for (int i = 0; i < actual_history; ++i)
                fens.push_back(fen_history[i]);
        } else {
            std::string fill = current_fen;
            for (int i = 0; i < 7; ++i) {
                fill = flip_side_to_move(fill);
                fens.push_front(fill);
            }
        }

        if (fens.size() > 7)
            fens.erase(fens.begin(), fens.begin() + (fens.size() - 7));

        std::deque<float> tensor(8 * 8 * 108, 0.0f);

        // === PROCESS HISTORICAL POSITIONS (0-6) ===
        for (int h = 0; h < 7; ++h) {
            std::istringstream fen_stream(fens[h]);
            std::string board_str, turn, castling, ep, halfmove, fullmove;
            fen_stream >> board_str >> turn >> castling >> ep >> halfmove >> fullmove;

            bool is_white_turn = (turn == "w");
            int base = h * 13;

            int rank = 7, file = 0;
            for (char c : board_str) {
                if (c == '/') {
                    rank--;
                    file = 0;
                } else if (isdigit(c)) {
                    file += (c - '0');
                } else {
                    bool piece_is_white = isupper(c);
                    char p = tolower(c);
                    int type = -1;
                    switch (p) {
                        case 'p': type = 0; break;
                        case 'n': type = 1; break;
                        case 'b': type = 2; break;
                        case 'r': type = 3; break;
                        case 'q': type = 4; break;
                        case 'k': type = 5; break;
                    }
                    if (type != -1) {
                        int tr = from_white_perspective ? rank : 7 - rank;
                        int tf = from_white_perspective ? file : 7 - file;
                        int channel = (piece_is_white == from_white_perspective) ? type : type + 6;
                        int idx = (tr * 8 + tf) * 108 + base + channel;
                        tensor[idx] = 1.0f;
                    }
                    file++;
                }
            }

            float color_val = (is_white_turn == from_white_perspective) ? 1.0f : 0.0f;
            for (int i = 0; i < 64; ++i)
                tensor[i * 108 + base + 12] = color_val;
        }

        // === CURRENT POSITION (7): 17 channels ===
        std::istringstream cs(current_fen);
        std::string board_str, turn, castling, ep, half, full;
        cs >> board_str >> turn >> castling >> ep >> half >> full;
        bool is_white_turn = (turn == "w");
        int base = 7 * 13;
        int rank = 7, file = 0;

        for (char c : board_str) {
            if (c == '/') {
                rank--;
                file = 0;
            } else if (isdigit(c)) {
                file += (c - '0');
            } else {
                bool piece_is_white = isupper(c);
                char p = tolower(c);
                int type = -1;
                switch (p) {
                    case 'p': type = 0; break;
                    case 'n': type = 1; break;
                    case 'b': type = 2; break;
                    case 'r': type = 3; break;
                    case 'q': type = 4; break;
                    case 'k': type = 5; break;
                }
                if (type != -1) {
                    int tr = from_white_perspective ? rank : 7 - rank;
                    int tf = from_white_perspective ? file : 7 - file;
                    int channel = (piece_is_white == from_white_perspective) ? type : type + 6;
                    int idx = (tr * 8 + tf) * 108 + base + channel;
                    tensor[idx] = 1.0f;
                }
                file++;
            }
        }

        bool wk = castling.find('K') != std::string::npos;
        bool wq = castling.find('Q') != std::string::npos;
        bool bk = castling.find('k') != std::string::npos;
        bool bq = castling.find('q') != std::string::npos;

        if (is_white_turn) {
            if (wk) for (int i = 0; i < 64; ++i) tensor[i * 108 + base + 13] = 1.0f;
            if (wq) for (int i = 0; i < 64; ++i) tensor[i * 108 + base + 14] = 1.0f;
            if (bk) for (int i = 0; i < 64; ++i) tensor[i * 108 + base + 15] = 1.0f;
            if (bq) for (int i = 0; i < 64; ++i) tensor[i * 108 + base + 16] = 1.0f;
        } else {
            if (bk) for (int i = 0; i < 64; ++i) tensor[i * 108 + base + 13] = 1.0f;
            if (bq) for (int i = 0; i < 64; ++i) tensor[i * 108 + base + 14] = 1.0f;
            if (wk) for (int i = 0; i < 64; ++i) tensor[i * 108 + base + 15] = 1.0f;
            if (wq) for (int i = 0; i < 64; ++i) tensor[i * 108 + base + 16] = 1.0f;
        }

        float color_val = (is_white_turn == from_white_perspective) ? 1.0f : 0.0f;
        for (int i = 0; i < 64; ++i)
            tensor[i * 108 + base + 12] = color_val;

        return {id, tensor};
    }
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