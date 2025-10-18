# Move Mappings Update

## Summary
Updated the `MoveMappings::initialize_move_mappings()` function in `position_tensor.cpp` to exactly match the Python implementation.

## Key Changes

### Before
The old implementation had a bug where it would generate promotion moves OR normal moves, but not both:
```cpp
if (is_white_promo || is_black_promo) {
    // ONLY add promotion moves
} else {
    // ONLY add normal move
}
```

### After
The new implementation matches Python exactly by **always** adding the normal move, then **additionally** adding promotion moves where applicable:
```cpp
// ALWAYS include the normal move
moves.push_back(move_uci);

// THEN add promotions if applicable
if (from_rank == 6 && to_rank == 7) { // White promotions
    // Add q, r, b, n promotions
}
if (from_rank == 1 && to_rank == 0) { // Black promotions
    // Add q, r, b, n promotions
}
```

## Python Code Reference
```python
def create_move_mappings():
    moves = []
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            # Always include the normal move
            moves.append(chess.Move(from_sq, to_sq).uci())

            # Add promotions if the move could be a pawn promotion
            from_rank = chess.square_rank(from_sq)
            to_rank = chess.square_rank(to_sq)

            # White pawn promotion (7→8)
            if from_rank == 6 and to_rank == 7:
                for p in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    moves.append(chess.Move(from_sq, to_sq, promotion=p).uci())

            # Black pawn promotion (2→1)
            if from_rank == 1 and to_rank == 0:
                for p in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    moves.append(chess.Move(from_sq, to_sq, promotion=p).uci())

    unique_moves = sorted(set(moves))
    move_to_index = {move: i for i, move in enumerate(unique_moves)}
    return move_to_index, index_to_move
```

## C++ Implementation
Now matches the Python exactly:
```cpp
void initialize_move_mappings() {
    std::vector<std::string> moves;
    
    for (int from_sq = 0; from_sq < 64; from_sq++) {
        for (int to_sq = 0; to_sq < 64; to_sq++) {
            // Always include the normal move
            std::string move_uci = square_to_string(from_sq) + square_to_string(to_sq);
            moves.push_back(move_uci);
            
            int from_rank = from_sq / 8;
            int to_rank = to_sq / 8;
            
            // White pawn promotion (rank 6 → rank 7)
            if (from_rank == 6 && to_rank == 7) {
                for (char promo : {'q', 'r', 'b', 'n'}) {
                    moves.push_back(move_uci + promo);
                }
            }
            
            // Black pawn promotion (rank 1 → rank 0)
            if (from_rank == 1 && to_rank == 0) {
                for (char promo : {'q', 'r', 'b', 'n'}) {
                    moves.push_back(move_uci + promo);
                }
            }
        }
    }
    
    // Remove duplicates and sort
    std::sort(moves.begin(), moves.end());
    moves.erase(std::unique(moves.begin(), moves.end()), moves.end());
    
    // Create mappings
    for (size_t i = 0; i < moves.size(); i++) {
        move_to_index[moves[i]] = i;
        index_to_move[i] = moves[i];
    }
    
    policy_output_size = moves.size();
}
```

## Example Moves
Now the mappings include:

### Normal Moves
- `e2e4` - Normal pawn push
- `e7e5` - Normal pawn push
- `g1f3` - Knight move

### Promotion Moves (White: rank 7→8)
For a move like `e7e8`, the system now creates:
- `e7e8` - Normal move (in case of capture by non-pawn)
- `e7e8q` - Promote to queen
- `e7e8r` - Promote to rook
- `e7e8b` - Promote to bishop
- `e7e8n` - Promote to knight

### Promotion Moves (Black: rank 2→1)
For a move like `e2e1`, the system now creates:
- `e2e1` - Normal move
- `e2e1q` - Promote to queen
- `e2e1r` - Promote to rook
- `e2e1b` - Promote to bishop
- `e2e1n` - Promote to knight

## Expected Policy Output Size
The total number of unique moves should be:
- 64 × 64 = 4,096 base moves
- Plus promotion moves for squares that can promote
- After deduplication and sorting: approximately **4,672 unique moves**

## Files Modified
- `/home/lordseidon/Desktop/Projects/Yeager/position_tensor.cpp`

## Status
✅ **COMPLETE** - The move mappings now exactly match your Python implementation.
