#include "fen_to_tensor.h"

// --- UTILITY AND PRINTING FUNCTION IMPLEMENTATIONS ---

float get_tensor_value(const std::deque<float>& tensor, int rank, int file, int channel) {
    if (rank < 0 || rank >= 8 || file < 0 || file >= 8 || channel < 0 || channel >= 18) {
        return 0.0f;
    }
    return tensor[(rank * 8 + file) * 18 + channel];
}

void print_tensor_summary(const std::deque<float>& tensor) {
    std::cout << "\nTensor summary (8x8x18):" << std::endl;
    for (int channel = 0; channel < 18; channel++) {
        int count = 0;
        for (int i = 0; i < 64; ++i) {
            if(tensor[i * 18 + channel] != 0.0f) count++;
        }
        if (count > 0) {
            std::string channel_name;
            const char* pieces[] = {"Pawn", "Knight", "Bishop", "Rook", "Queen", "King"};
            if (channel < 6) { channel_name = std::string("Own ") + pieces[channel]; }
            else if (channel < 12) { channel_name = std::string("Opponent ") + pieces[channel - 6]; }
            else if (channel == 12) channel_name = "Own Kingside Castling";
            else if (channel == 13) channel_name = "Own Queenside Castling";
            else if (channel == 14) channel_name = "Opponent Kingside Castling";
            else if (channel == 15) channel_name = "Opponent Queenside Castling";
            else if (channel == 16) channel_name = "En Passant";
            else if (channel == 17) channel_name = "Player Color";
            std::cout << "  Channel " << channel << " (" << channel_name << "): " << count << " non-zero values" << std::endl;
        }
    }
}

void print_tensor_channel(const std::deque<float>& tensor, int channel) {
    std::cout << "\nChannel " << channel << " values:" << std::endl;
    std::cout << "   a  b  c  d  e  f  g  h" << std::endl;
    std::cout << "  ------------------------" << std::endl;
    for (int rank = 7; rank >= 0; rank--) {
        std::cout << (rank + 1) << "|";
        for (int file = 0; file < 8; file++) {
            float value = get_tensor_value(tensor, rank, file, channel);
            std::cout << " " << (value > 0.5f ? "1" : ".") << " ";
        }
        std::cout << "|" << std::endl;
    }
    std::cout << "  ------------------------" << std::endl;
}

void print_tensor_detailed(const std::deque<float>& tensor) {
    std::cout << "\nDetailed Tensor Analysis:" << std::endl;
    const char* piece_names[] = {"Pawn", "Knight", "Bishop", "Rook", "Queen", "King"};
    
    for (int piece_type = 0; piece_type < 6; piece_type++) {
        int own_channel = piece_type;
        int opp_channel = piece_type + 6;
        bool own_has_pieces = false, opp_has_pieces = false;
        
        for (int sq = 0; sq < 64; sq++) {
            if (tensor[sq * 18 + own_channel] > 0.5f) own_has_pieces = true;
            if (tensor[sq * 18 + opp_channel] > 0.5f) opp_has_pieces = true;
        }
        
        if (own_has_pieces) {
            std::cout << "\nOwn " << piece_names[piece_type] << "s (Channel " << own_channel << "):";
            print_tensor_channel(tensor, own_channel);
        }
        if (opp_has_pieces) {
            std::cout << "\nOpponent " << piece_names[piece_type] << "s (Channel " << opp_channel << "):";
            print_tensor_channel(tensor, opp_channel);
        }
    }
    
    std::cout << "\n--- METADATA ---" << std::endl;
    print_tensor_summary(tensor);
    
    float color_value = get_tensor_value(tensor, 0, 0, 17);
    std::cout << "\nPlayer Color (Channel 17): " << color_value << " (1.0 = WHITE, 0.0 = BLACK)" << std::endl;
}

void print_raw_tensor(const std::deque<float>& tensor) {
    std::cout << "\n--- Raw Tensor Data (1152 floats) ---" << std::endl;
    std::cout << "[";
    for (size_t i = 0; i < tensor.size(); ++i) {
        std::cout << tensor[i];
        if (i < tensor.size() - 1) {
            std::cout << ", ";
        }
        if ((i + 1) % 18 == 0) {
            std::cout << "\n ";
        }
    }
    std::cout << "]" << std::endl;
}

void print_full_tensor_structured(const std::deque<float>& tensor) {
    std::cout << "\n\n--- Full Structured Tensor Dump (8x8x18) ---" << std::endl;
    const char* channel_names[] = {
        "Own Pawns", "Own Knights", "Own Bishops", "Own Rooks", "Own Queens", "Own Kings",
        "Opponent Pawns", "Opponent Knights", "Opponent Bishops", "Opponent Rooks", "Opponent Queens", "Opponent Kings",
        "Own Kingside Castle", "Own Queenside Castle", "Opponent Kingside Castle", "Opponent Queenside Castle",
        "En Passant Target", "Player Color (1=W, 0=B)"
    };
    for (int channel = 0; channel < 18; ++channel) {
        std::cout << "\n\n--- Channel " << channel << ": " << channel_names[channel] << " ---" << std::endl;
        std::cout << "     a    b    c    d    e    f    g    h" << std::endl;
        std::cout << "    ---------------------------------------" << std::endl;
        for (int rank = 7; rank >= 0; --rank) {
            std::cout << " " << (rank + 1) << " |";
            for (int file = 0; file < 8; ++file) {
                float value = get_tensor_value(tensor, rank, file, channel);
                std::cout << " " << std::fixed << std::setprecision(1) << value << " ";
            }
            std::cout << "|" << std::endl;
        }
        std::cout << "    ---------------------------------------" << std::endl;
    }
}