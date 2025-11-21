import numpy as np


def generate_random_board(n: int, b: int, x: int, y: int) -> np.ndarray:
    """
    Generate a random Minesweeper board of size nxn with b bombs.
    
    The board format matches the game's internal representation:
    - -2 (BrickType.MINE) for cells containing mines
    - 0 (BrickType.EMPTY) for empty cells with no neighboring mines
    - 1-8 for cells with that many neighboring mines
    
    Args:
        n: Size of the board (nxn)
        b: Number of bombs to place
        x: Row index of the first position (0-indexed)
        y: Column index of the first position (0-indexed)
    
    Returns:
        board: numpy array of shape (n, n) with dtype=np.int32
               containing the board state with mines and numbers
    """
    if b >= n * n:
        raise ValueError(f"Cannot place {b} bombs on a {n}x{n} board (only {n*n} cells available)")
    
    if not (0 <= x < n and 0 <= y < n):
        raise ValueError(f"First position ({x}, {y}) is out of bounds for a {n}x{n} board")
    
    # Initialize board 
    board = np.zeros((n, n), dtype=np.int32)
    
    # Create list of all possible positions (excluding first position)
    all_positions = [(i, j) for i in range(n) for j in range(n) if (i, j) != (x, y)]
    bomb_indices = np.random.choice(len(all_positions), size=b, replace=False)
    for idx in bomb_indices:
        i, j = all_positions[idx]
        board[i, j] = -2 
    for i in range(n):
        for j in range(n):
            if board[i, j] != -2:  
                neighbor_mines = 0
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni, nj = i + di, j + dj
                        if (di, dj) == (0, 0):
                            continue
                        if 0 <= ni < n and 0 <= nj < n:
                            if board[ni, nj] == -2:  
                                neighbor_mines += 1
                board[i, j] = neighbor_mines
    
    return board

