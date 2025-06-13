# feed in a known board, assert solve(...) returns correct solution

import pytest
from sudoku_solver import solve
from utils import load_sudoku_samples


def test_solve_samples():
    samples = load_sudoku_samples("data/samples_sudoku.json")
    assert samples, "No samples loaded"
    for puzzle, solution in samples:
        board = [row.copy() for row in puzzle]
        solved = solve(board)
        assert solved is not None, "returned None for a valid puzzle"
        assert solved == solution, "output does not match known solution"


def test_invalid_board():
    # a board with duplicate '1' in first row
    board = [["1"] * 9] + [["."] * 9 for _ in range(8)]
    assert solve(board) is None

