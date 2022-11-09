#include <iostream>
#include "position.h"
#include "tables.h"
// #include "piece_tables.h"
#include <string>
#include <ostream>
#include "types.h"
#include "helpers.h"
#include "piece_advantage.h"

int checkmate = -1e7;

bool is_checkmate(Position board)  {

    if (board.turn() == 0) {
        int cnt = MoveList<WHITE>(board).size();
        if (cnt == 0) {
            if (board.in_check<WHITE>()) {
                return true;
            }
        }
        }  

    else {
        int cnt = MoveList<BLACK>(board).size();
        if (cnt == 0) {
            if (board.in_check<BLACK>()) {
                return true;
            }
        }
        }
    return false; 
}

bool is_stalemate(Position board)  {
    if (board.turn() == 0) {
        int cnt = MoveList<WHITE>(board).size();
        if (cnt == 0) {
            if (!board.in_check<WHITE>()) {
                return true;
            }
        }
        }  

    else {
        int cnt = MoveList<BLACK>(board).size();
        if (cnt == 0) {
            if (!board.in_check<BLACK>()) {
                return true;
            }
        }
        }
    return false; 
}


int evaluator (Position board) {
    // white section 

    std::pair<int, int> wpawnn  = pawn_advantage(board.pawns<WHITE>(), true);
    int Wpawn = wpawnn.first;
    int wpawn = wpawnn.second;
    
    
    std::pair<int, int> ewpawnn = pawn_end_advantage(board.pawns<WHITE>(), true);  
    int eWpawn = ewpawnn.first;
    int ewpawn = ewpawnn.second;
    
    
    std::pair<int, int> wb = bishop_advantage(board.bishops<WHITE>(), true);  
    int Wbishop = wb.first;
    int wbishop = wb.second;
    
    std::pair<int, int> wkn = knight_advantage(board.knights<WHITE>(), true);  
    int Wknight = wkn.first;
    int wknight = wkn.second;
    
    std::pair<int, int> wrk = rook_advantage(board.rooks<WHITE>(), true);  
    int Wrook = wrk.first;
    int wrook = wrk.second;
    
    std::pair<int, int> wqn = queen_advantage(board.queens<WHITE>(), true);  
    int Wqueen = wqn.first;
    int wqueen = wqn.second;
    
    int Wking = king_mid_advantage(board.kings<WHITE>(), true);  
    int WKingE = king_end_advantage(board.kings<WHITE>(), true);  


    // black section 
    // endgames 
     
    std::pair<int, int> pawnn = pawn_advantage(board.pawns<BLACK>(), false);  
    int Bpawn = pawnn.first;
    int bpawn = pawnn.second;
    // std::cout<< "Bpiece: -> "<< board.pawns<BLACK>() << std::endl;
    
    
    std::pair<int, int> epawnn = pawn_end_advantage(board.pawns<BLACK>(), false);  
    int eBpawn = epawnn.first;
    int ebpawn = epawnn.second;
    
    
    std::pair<int, int> bb = bishop_advantage(board.bishops<BLACK>(), false);  
    int Bbishop = bb.first;
    int bbishop = bb.second;
    
    std::pair<int, int> kn = knight_advantage(board.knights<BLACK>(), false);  
    int Bknight = kn.first;
    int bknight = kn.second;
    
    std::pair<int, int> rook = rook_advantage(board.rooks<BLACK>(), false);  
    int Brook = rook.first;
    int brook = rook.second;
    
    std::pair<int, int> qn = queen_advantage(board.queens<BLACK>(), false);  
    int Bqueen = qn.first;
    int bqueen = qn.second;
    
    int Bking = king_mid_advantage(board.kings<BLACK>(), false);  
    int BkingE = king_end_advantage(board.kings<BLACK>(), false);  
    
    // std::cout<< "Pos pawn  : -> "<< Wpawn - Bpawn<< std::endl;
    

    int material_advantage = 100 * (wpawn - bpawn)
    + (300 * (wknight - bknight))
    + (360 * (wbishop - bbishop))
    + (500 * (wrook - brook))
    + (900 * (wqueen - bqueen));

    int positional_advantage = (Wpawn - Bpawn)
    + (Wknight - Bknight)
    + (Wbishop - Bbishop)
    + (Wrook - Brook)
    + (Wqueen - Bqueen)
    + (Wking - Bking);

    // std::cout<<"Material Advantage:  "<< material_advantage << std::endl; 
    // std::cout<<"\n Positional Advantage: -> "<< positional_advantage<<std::endl;
    return positional_advantage + material_advantage;
}

