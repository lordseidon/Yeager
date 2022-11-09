#include <iostream>
#include "piece_tables.h"
#include "position.h"


std::pair<unsigned long long, int> pawn_advantage(unsigned long long Bpiece, bool color) {
    int length = 0;
    int pawn_add = 0;
    int piece_array [8] = {};
    int addL = 0;

    if (color) {   
        while (Bpiece >= 1) {
            if (Bpiece %2 != 0) {
                piece_array[addL] = length;
                addL++;      
        }
        length++;
        Bpiece = Bpiece / 2;
    }

    for (int i=0; i<addL; i++) {
        pawn_add += pawn_tb[(63 - piece_array[i])/8][7 - ( (63 - piece_array[i]) %8)];
        // std::cout<< "White pawn advantage: -> "<< pawn_add << std::endl;
  }
    }
    else {
        while (Bpiece >= 1) {
            if (Bpiece %2 != 0) {
                    piece_array[addL] = length;
                    addL++;      
            }
            length++;
            Bpiece = Bpiece / 2;
            }
     
        for (int i=0; i<addL; i++) {
            pawn_add += pawn_tb[piece_array[i]/8][piece_array[i] %8];
            // std::cout<< "Black pawn advantage   : -> "<< piece_array[i] << std::endl;
        }        
        
    }
    // std::cout<<"Pos pawn god white " << pawn_add << " True " << color<< std::endl;
    return std::make_pair(pawn_add, addL);
}

std::pair<int, int> bishop_advantage(unsigned long long Bpiece, bool color) {
    int length = 0;
    int pawn_add = 0;
    int piece_array [8] = {};
    int addL = 0;
    if (color) {     
        while (Bpiece >= 1) {
            if (Bpiece %2 != 0) {
                piece_array[addL] = length;
                addL++;      
        }
        length++;
        Bpiece = Bpiece / 2;
    }

    for (int i=0; i<addL; i++) {
        pawn_add += bishop_tb[(63 - piece_array[i])/8][7 - ( (63 - piece_array[i]) %8)];
    }        
        
  }

 else {
    while (Bpiece >= 1) {
            if (Bpiece %2 != 0) {
                piece_array[addL] = length;
                addL++;      
        }
        length++;
        Bpiece = Bpiece / 2;
    }

    for (int i=0; i<addL; i++) {
        pawn_add += bishop_tb[piece_array[i]/8][piece_array[i] %8];
    }        
        
  }
  return std::make_pair(pawn_add, addL);
}

std::pair<int, int> knight_advantage(unsigned long long Bpiece, bool color) {
    int length = 0;
    int pawn_add = 0;
    int piece_array [8] = {};
    int addL = 0;
    if (color) {     
        while (Bpiece >= 1) {
            if (Bpiece %2 != 0) {
                piece_array[addL] = length;
                addL++;      
        }
        length++;
        Bpiece = Bpiece / 2;
    }

    for (int i=0; i<addL; i++) {
        pawn_add += knight_tb[(63 - piece_array[i])/8][7 - ( (63 - piece_array[i]) %8)];
    }        
        // std::cout<< "White pawn advantage: -> "<< pawn_add << std::endl;
  }

  else {
    while (Bpiece >= 1) {
            if (Bpiece %2 != 0) {
                piece_array[addL] = length;
                addL++;      
        }
        length++;
        Bpiece = Bpiece / 2;
    }

    for (int i=0; i<addL; i++) {
        pawn_add += knight_tb[piece_array[i]/8][piece_array[i] %8];
    }        
        // std::cout<< "Black pawn advantage: -> "<< pawn_add << std::endl;
  }
    return std::make_pair(pawn_add, addL);
}

std::pair<int, int> rook_advantage(unsigned long long Bpiece, bool color) {
    int length = 0;
    int pawn_add = 0;
    int piece_array [8] = {};
    int addL = 0;
    if (color) {     
        while (Bpiece >= 1) {
            if (Bpiece %2 != 0) {
                piece_array[addL] = length;
                addL++;      
        }
        length++;
        Bpiece = Bpiece / 2;
    }

    for (int i=0; i<addL; i++) {
        pawn_add += rook_tb[(63 - piece_array[i])/8][7 - ( (63 - piece_array[i]) %8)];
    }        
        // std::cout<< "White pawn advantage: -> "<< pawn_add << std::endl;
  }

  else {
    while (Bpiece >= 1) {
            if (Bpiece %2 != 0) {
                piece_array[addL] = length;
                addL++;      
        }
        length++;
        Bpiece = Bpiece / 2;
    }

    for (int i=0; i<addL; i++) {
        pawn_add += rook_tb[piece_array[i]/8][piece_array[i] %8];
    }        
        // std::cout<< "Black pawn advantage: -> "<< pawn_add << std::endl;
  }
    return std::make_pair(pawn_add, addL);
}


std::pair<int, int> queen_advantage(unsigned long long Bpiece, bool color) {
    int length = 0;
    int pawn_add = 0;
    int piece_array [8] = {};
    int addL = 0;
    if (color) {     
        while (Bpiece >= 1) {
            if (Bpiece %2 != 0) {
                piece_array[addL] = length;
                addL++;      
        }
        length++;
        Bpiece = Bpiece / 2;
    }

    for (int i=0; i<addL; i++) {
        pawn_add += queen_tb[(63 - piece_array[i])/8][7 - ( (63 - piece_array[i]) %8)];
    }        
        // std::cout<< "White pawn advantage: -> "<< pawn_add << std::endl;
  }

  else {
    while (Bpiece >= 1) {
            if (Bpiece %2 != 0) {
                piece_array[addL] = length;
                addL++;      
        }
        length++;
        Bpiece = Bpiece / 2;
    }

    for (int i=0; i<addL; i++) {
        pawn_add += queen_tb[piece_array[i]/8][piece_array[i] %8];
    }        
        // std::cout<< "Black pawn advantage: -> "<< pawn_add << std::endl;
  }
    return std::make_pair(pawn_add, addL);
}

int king_mid_advantage(unsigned long long Bpiece, bool color) {
    int length = 0;
    int pawn_add = 0;
    int piece_array [8] = {};
    int addL = 0;
    if (color) {     
        while (Bpiece >= 1) {
            if (Bpiece %2 != 0) {
                piece_array[addL] = length;
                addL++;      
        }
        length++;
        Bpiece = Bpiece / 2;
    }

    for (int i=0; i<addL; i++) {
        pawn_add += king_middle[(63 - piece_array[i])/8][7 - ( (63 - piece_array[i]) %8)];
    }        
        // std::cout<< "White pawn advantage: -> "<< pawn_add << std::endl;
  }

  else {
    while (Bpiece >= 1) {
            if (Bpiece %2 != 0) {
                piece_array[addL] = length;
                addL++;      
        }
        length++;
        Bpiece = Bpiece / 2;
    }

    for (int i=0; i<addL; i++) {
        pawn_add += king_middle[piece_array[i]/8][piece_array[i] %8];
    }        
        // std::cout<< "Black pawn advantage: -> "<< pawn_add << std::endl;
  }
    return pawn_add;
}


// Endgames Endgames Endgames
int king_end_advantage(unsigned long long Bpiece, bool color) {
    int length = 0;
    int pawn_add = 0;
    int piece_array [8] = {};
    int addL = 0;
    if (color) {     
        while (Bpiece >= 1) {
            if (Bpiece %2 != 0) {
                piece_array[addL] = length;
                addL++;      
        }
        length++;
        Bpiece = Bpiece / 2;
    }

    for (int i=0; i<addL; i++) {
        pawn_add += king_ending[(63 - piece_array[i])/8][7 - ( (63 - piece_array[i]) %8)];
    }        
        // std::cout<< "White pawn advantage: -> "<< pawn_add << std::endl;
  }

  else {
    while (Bpiece >= 1) {
            if (Bpiece %2 != 0) {
                piece_array[addL] = length;
                addL++;      
        }
        length++;
        Bpiece = Bpiece / 2;
    }

    for (int i=0; i<addL; i++) {
        pawn_add += king_ending[piece_array[i]/8][piece_array[i] %8];
    }        
        // std::cout<< "Black pawn advantage: -> "<< pawn_add << std::endl;
  }
    return pawn_add;
}

std::pair<int, int> pawn_end_advantage(unsigned long long Bpiece, bool color) {
    int length = 0;
    int pawn_add = 0;
    int piece_array [8] = {};
    int addL = 0;
    if (color) {     
        while (Bpiece >= 1) {
            if (Bpiece %2 != 0) {
                piece_array[addL] = length;
                addL++;      
        }
        length++;
        Bpiece = Bpiece / 2;
    }

    for (int i=0; i<addL; i++) {
        pawn_add += pawn_ending[(63 - piece_array[i])/8][7 - ( (63 - piece_array[i]) %8)];
    }        
        // std::cout<< "White pawn advantage: -> "<< pawn_add << std::endl;
  }

  else {
    while (Bpiece >= 1) {
            if (Bpiece %2 != 0) {
                piece_array[addL] = length;
                addL++;      
        }
        length++;
        Bpiece = Bpiece / 2;
    }

    for (int i=0; i<addL; i++) {
        pawn_add += pawn_ending[piece_array[i]/8][piece_array[i] %8];
    }        
        // std::cout<< "Black pawn advantage: -> "<< pawn_add << std::endl;
  }
    return std::make_pair(pawn_add, addL);
}

