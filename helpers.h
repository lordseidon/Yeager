#include <iostream>

// function to get the number of a particular type of piece on the board

int piece_value(unsigned long long Bpiece) {
  int length = 0;
  int piece [8] = {};
  int arrL = 0;

  while (Bpiece >= 1) {
    if (Bpiece %2 != 0) {
      piece[arrL] = length;
      arrL++;
    }
  Bpiece = Bpiece / 2;
  }
  std::cout<< "Array: "<< *piece << " Arrl: " << arrL<< std::endl;
  return *piece;
}
// ================================== =====================================

