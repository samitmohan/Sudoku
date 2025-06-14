from typing import List
import numpy as np
import json


def probs_to_board(probs: List[List[float]], blank_index: int = 0) -> List[List[str]]:
    # converts flat list of 81 prob vectors into 9*91 sudoku board of strings
    if len(probs) != 81:
        raise ValueError(f"Expected 81 probability vectors got {len(probs)} vectors")
    flat = []
    for p in probs:
        index = int(np.argmax(p))
        flat.append("." if index == blank_index else str(index))

    board = [flat[r * 9 : (r + 1) * 9] for r in range(9)]
    return board


def show_board(board: List[List[str]]) -> None:
    for r in range(9):
        row = board[r]
        print(" ".join(row))
    print()


def load_sudoku_samples(path: str):
    with open(path, "r") as f:
        data = json.load(f)

    samples = []
    for entry in data:
        puz = entry["puzzle"]
        sol = entry["solution"]

        # Convert '0' → '.' in the puzzle rows
        puzzle_board = [[ch if ch != "0" else "." for ch in row] for row in puz]
        solution_board = [list(row) for row in sol]

        if len(puzzle_board) != 9 or len(solution_board) != 9:
            raise ValueError("Each puzzle/solution must have 9 rows.")
        samples.append((puzzle_board, solution_board))

    return samples
