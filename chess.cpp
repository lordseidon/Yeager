#include <iostream>
#include "position.h"
#include "tables.h"
#include <string>
#include "search.h"
// #include "move_order.h"
#include <map>
#include <ostream>
#include <bits/stdc++.h>
#include "types.h"
#include <chrono>

// compile command >
// g++ -o out.o chess.cpp position.cpp tables.cpp types.cpp -o chess.exe

int main() {

  initialise_all_databases();
	zobrist::initialise_zobrist_keys();

  
    Position p;
    // Position::set("8/1k6/8/8/7q/4q1q1/6K1/4q3 w - - 1 6", p); // checkmate position black
    // Position::set("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4", p); // checkmate position white
    // Position::set("8/1n1b4/8/8/8/8/5k2/7K w - - 10 23", p); // mate in position white
    // Position::set("8/8/2k5/8/5q1q/4q3/6K1/4q3 w - - 1 6", p); //stalemate position
    Position::set("rnb2rk1/ppp2p1p/5B2/4P3/2B5/2b2R2/P1P3PP/R6K b - - 1 16", p);
    // Position::set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", p);

    std::cout << p; 
    MoveListCaptures<WHITE> list(p);
    // MoveList<WHITE> list(p);

    std::cout<<"Move List : " <<MoveList<WHITE>(p).size()<<'\n';
    std::cout<<"Move List Captures: " <<MoveListCaptures<WHITE>(p).size()<<'\n';


    // for (int i=0; i<MoveList<BLACK>(p).size(); i++) {
      // if(MoveList<BLACK>(p).begs()[i].is_capture()) {
        // std::cout<<"Capture "<<MoveList<BLACK>(p).begin()[i]<<std::endl;
      // } 
    // }
    
    // for (int i=0; i<movesize; i++) {
    //   // std::cout<<"Fake moves "<<i << " "<<fake_bishops[i]<<std::endl;
    // }

    
    
    int counter = 0;
    // std::cout<<"Length of moves object = "<<p.in_check<WHITE>()<<'\n';
    // Move movelist [MoveListCaptures<WHITE>(p).size()];
    // int idx = 0;
    // for(Move m : list) {
    //   movelist[idx] = m;
    //     std::cout << "MoveU "<< counter++ << " "<< m << std::endl;
    //     std::cout << "Move from "<< " "<<p.at(m.from())<<std::endl;
    //     std::cout << "Move to "<< " "<<p.at(m.to())<<std::endl;
    //     std::cout << "Move to "<< " "<<MVV_LVA[p.at(m.to())][p.at(m.from())]<<std::endl;
    //     idx++;
    // }
    // quickSort(movelist, 0,MoveListCaptures<WHITE>(p).size()-1, p);
    // for (int i=0; i<MoveListCaptures<WHITE>(p).size(); i++) {
    //   std::cout<<"Capture Move "<<movelist[i]<<endl;
    // }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::cout<<"Negamax Score = "<<NegaMax(p, maxdepth, -Checkmate, Checkmate, !p.turn())<<std::endl;
    // std::cout<<"Negamax Score = "<<NegaScore(p, maxdepth, !p.turn())<<std::endl;
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    auto diff = end - begin;
    std::cout << "Time difference = "
		<< std::chrono::duration_cast<std::chrono::microseconds>(diff).count() << " [microseconds]\n";


    std::cout<<"Static board eval : "<<evaluator(p)<<std::endl; 
    std::cout<<"Stalemate "<<is_stalemate(p)<<'\n';
    std::cout<<"Checkmate "<<is_checkmate(p)<<'\n';
    std::cout<<"Nodes: "<<nodes<<std::endl; 
    std::cout<<"QNodes: "<<qnodes<<std::endl; 

    // for (auto itr=pv_table.begin(); itr!=pv_table.end(); ++itr) {
    //   std::cout<<"Move : "<<itr->first;
    //   std::cout<<" Score : "<<itr->second<<endl;
    // }
    
    return 0;
  }
