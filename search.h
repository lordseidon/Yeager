#include <iostream>
#include <string>
#include "evaluator.h"
#include <map>
#include <ostream>
using namespace std;

int nodes = 0;
int qnodes = 0;
int Checkmate = 1e7;
int maxdepth = 6;
int maxCaptures = 1;
bool isNull = true;
const int MAX_PLY = 30;

Move bestMove;
Move plyone;
Move plytwo;
Move plythree;

int ply = 1;
map<int, int> pv_table;

Move pv_table_lines;

int nullMove(Position board, int alpha, int beta, bool white) {
    return 0;
}

int sort_captures() {
    return 0;
}

int quiet_move(Position board, int depth, int alpha, int beta, bool white) {

    if (depth == 0) {
        return evaluator(board);
    }
    if (white) {
       if (MoveListCaptures<WHITE>(board).size() == 0) {
        return evaluator(board);
       } 
    }
    else {
        if (MoveListCaptures<BLACK>(board).size() == 0) {
        return evaluator(board);
       } 
    }
    qnodes++;
    int eval = evaluator(board);
    if (is_checkmate(board)) {
           return board.turn() ==true ? -checkmate:
            checkmate;
    }
    if (is_stalemate(board)) {
        return 0;
    } 
    if (white) {
        int best_score = -Checkmate;
        MoveListCaptures<WHITE> list(board);
        for(Move m : list) {
            board.play<WHITE>(m);
            int score = quiet_move(board, depth-1, alpha, beta, !white);
             
           board.undo<WHITE>(m);  
           alpha = std::max(alpha, score);

            if (alpha >= beta) {
                break;
            }
        }
        return alpha; 
    }
    else {
        int minScore = Checkmate;
        MoveListCaptures<BLACK> list(board);
        for(Move m : list) {
            board.play<BLACK>(m);
            int score = quiet_move(board, depth-1, alpha, beta, !white);  
            board.undo<BLACK>(m); 
            
            beta = std::min(beta, score);

            if (alpha >= beta) {
                break;
            }
            }
        return beta; 
    } 
}

int NegaMax(Position board, int depth, int alpha, int beta, bool white) {
    nodes++;
    if (depth == 0) {
        // return evaluator(board);
        return quiet_move(board, maxCaptures, alpha, beta, white);
    }
    if (is_checkmate(board)) {
            return board.turn() ==true ? -checkmate:
            checkmate;
    }
    if (is_stalemate(board)) {
        return 0;
    }  

    if (white) {
        int best_score = -Checkmate;
        MoveList<WHITE> list(board);

        int idx=0;
        for(Move m : list) {
            board.play<WHITE>(m);
            int score = NegaMax(board, depth-1, alpha, beta, !white);
        
            if (score > best_score) {
                best_score =  score; 
                if (depth == maxdepth) {
                    bestMove = m;
                    std::cout<<"Best move "<<bestMove<<" Score "<<best_score<<" Nodes "<<nodes<<" QNodes "<<qnodes<<std::endl;
                }
            }
           board.undo<WHITE>(m);  

            if (best_score > alpha) {
                alpha = best_score;
            }
            idx++;
            if (alpha >= beta) {
                break;
            }
        }
        return best_score; 
    }
    else {
        int minScore = Checkmate;
        MoveList<BLACK> list(board);
        int idx=0;
        for(Move m : list) {
            board.play<BLACK>(m);
            int score = NegaMax(board, depth-1, alpha, beta, !white);  
            if (score < minScore) {
                minScore = score;
                if (depth == maxdepth) {
                    bestMove = m;
                    std::cout<<"Best move min "<<bestMove<<" Score "<<minScore<<" Nodes "<<nodes<<" QNodes "<<qnodes<<std::endl;
                }
            }
            board.undo<BLACK>(m); 
            if (depth == maxdepth) {
                   pv_table[idx] = score;
                //    std::cout<<"move "<<idx<<" "<<m<<" Score "<<score<<" Nodes "<<nodes<<" QNodes "<<qnodes<<std::endl;
                }
            if (minScore < beta) {
                beta = minScore;
                
            }
            idx++;

            if (alpha >= beta) {
                break;
            }
        } 
        return minScore; 
    } 
    
    return 0;
}



int NegaScore(Position board, int depth, bool white) {
    // nodes++;

    if (white) {
        MoveList<WHITE> list(board);

        int idx=0;
        for(Move m : list) {
            board.play<WHITE>(m);
            int score = NegaMax(board, depth, -Checkmate, Checkmate, !white);
            if (depth == maxdepth) {
                pv_table[idx] = score;
                std::cout<<"move "<<idx<<" "<<m<<" Score "<<score<<" Nodes "<<nodes<<" QNodes "<<qnodes<<std::endl;
                }
            board.undo<WHITE>(m);  
            idx++;
        }
    }
    else {
        MoveList<BLACK> list(board);
        int idx=0;
        for(Move m : list) {
            board.play<BLACK>(m);
            int score = NegaMax(board, depth, -Checkmate, Checkmate, !white);
            if (depth == maxdepth) {
                pv_table[idx] = score;
                std::cout<<"move "<<idx<<" "<<m<<" Score "<<score<<" Nodes "<<nodes<<" QNodes "<<qnodes<<std::endl;
                }
            board.undo<BLACK>(m); 
            idx++;
        } 
    }     
    return 0;
}


