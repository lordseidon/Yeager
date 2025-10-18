# Position History Implementation

## Overview
This implementation adds position history tracking to the chess engine, allowing the neural network to use the last 7 positions plus the current position (8 total) for evaluation.

## Changes Made

### 1. fen_to_tensor.h
**New Functions:**
- `fen_to_board_tensor()`: Helper function to convert a single FEN to a board tensor with 13 channels (12 pieces + side to move)
- `fen_history_to_tensor()`: Main function that takes 8 FEN strings and produces a combined tensor

**Tensor Structure:**
- **Historical positions (0-6)**: Each has 13 channels
  - Channels 0-5: Own pieces (P, N, B, R, Q, K)
  - Channels 6-11: Opponent pieces (P, N, B, R, Q, K)
  - Channel 12: Side to move (1.0 for white, 0.0 for black)
  
- **Current position (7)**: Has 17 channels
  - Channels 0-5: Own pieces (P, N, B, R, Q, K)
  - Channels 6-11: Opponent pieces (P, N, B, R, Q, K)
  - Channel 12: Own kingside castling
  - Channel 13: Own queenside castling
  - Channel 14: Opponent kingside castling
  - Channel 15: Opponent queenside castling
  - Channel 16: Side to move (1.0 for white, 0.0 for black)

**Total Tensor Size:**
- 7 historical positions × 64 squares × 13 channels = 5,824 values
- 1 current position × 64 squares × 17 channels = 1,088 values
- **Total: 6,912 values (8×8×108)**

**Features:**
- Automatically pads with empty positions if less than 8 positions are provided
- All positions are viewed from the perspective of the current player
- Backward compatible: old `fen_to_tensor()` function still works with 18 channels

### 2. mcts.h
**Changes to MCTSNode:**
- Added `std::deque<std::string> fen_history` field to track the 7 previous positions leading to each node

**Changes to MCTS:**
- Added overload: `Move run_search(const Position& initial_pos, int iterations, bool clear_after_search, const std::deque<std::string>& position_history)`

### 3. mcts.cpp
**Changes to convert_position_to_tensor():**
- Now takes `const std::deque<std::string>& fen_history` parameter
- Uses `fen_history_to_tensor()` instead of `fen_to_tensor()`

**Changes to run_search():**
- Created wrapper that calls the new overload with empty history (backward compatible)
- New overload accepts and uses position history
- Initializes root node with the provided history

**Changes to search_worker():**
- Tracks FEN history as it traverses the tree
- Before each move, saves the current position to history
- Maintains only the last 7 positions
- Stores history in each node when expanding

### 4. game.cpp
**Changes to play_single_game():**
- Added `std::deque<std::string> fen_history` to track game history
- Before each move, saves current position to history
- Passes history to MCTS via `run_search()`
- Maintains only last 7 positions (pops front when size > 7)

## Usage Example

```cpp
// In game loop
std::deque<std::string> fen_history;

while (game_in_progress) {
    // Run MCTS search with position history
    Move best_move = mcts_engine.run_search(pos, iterations, false, fen_history);
    
    // Save current position before making move
    fen_history.push_back(pos.fen());
    if (fen_history.size() > 7) {
        fen_history.pop_front(); // Keep only last 7
    }
    
    // Make the move
    pos.play(best_move);
}
```

## Neural Network Input Shape

Your neural network should expect input with shape:
- **Shape**: `(batch_size, 8, 8, 108)`
- **Channels**: 108 (7×13 + 17)

## Testing

Run the test program to verify the implementation:
```bash
./test_history_tensor
```

All tests should pass, verifying:
1. Correct tensor size (6912 values)
2. Proper handling of full history (8 positions)
3. Correct padding for partial history
4. Backward compatibility with old function

## Benefits

1. **Temporal Information**: Network can now see move sequences and tactical patterns
2. **Repetition Detection**: Network can learn about repetitive positions
3. **Better Evaluation**: Historical context improves position understanding
4. **AlphaZero-style**: Similar to AlphaZero's 8-position history approach

## Notes

- Empty positions (at game start) are represented as all zeros
- All positions are rotated to be from the current player's perspective
- Castling rights are only included for the current position
- En passant information was removed (can be inferred from position history)
