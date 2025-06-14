from typing import List, Tuple, Set

DIGITS: List[str] = [str(i) for i in range(1, 10)]


def initialize_board(board):
    rows: List[Set[str]] = [set() for _ in range(9)]
    cols: List[Set[str]] = [set() for _ in range(9)]
    boxes: List[Set[str]] = [set() for _ in range(9)]
    empties: List[Tuple[int, int]] = []

    for r in range(9):
        for c in range(9):
            val = board[r][c]
            if val == ".":
                empties.append((r, c))
            else:
                b_index = (r // 3) * 3 + (c // 3)
                if val in rows[r] or val in cols[c] or val in boxes[b_index]:
                    # duplicate in row/col/box → invalid board
                    return None
                rows[r].add(val)
                cols[c].add(val)
                boxes[b_index].add(val)

    return rows, cols, boxes, empties


def candidates(
    r: int,
    c: int,
    rows: List[Set[str]],
    cols: List[Set[str]],
    boxes: List[Set[str]],
) -> List[str]:
    """
    Return all digits that can legally go in board[r][c],
    given the current rows/cols/boxes constraints.
    """
    b_index = (r // 3) * 3 + (c // 3)
    used = rows[r] | cols[c] | boxes[b_index]
    return [d for d in DIGITS if d not in used]


def solve(board: List[List[str]], *, verbose: bool = False):
    init = initialize_board(board)
    if init is None:
        return None
    rows, cols, boxes, empties = init

    # 2) MRV: sort empties by fewest candidates first
    empties.sort(key=lambda rc: len(candidates(rc[0], rc[1], rows, cols, boxes)))

    def backtrack(idx: int) -> bool:
        if idx == len(empties):
            return True  # all filled

        r, c = empties[idx]
        b_index = (r // 3) * 3 + (c // 3)
        for d in candidates(r, c, rows, cols, boxes):
            if verbose:
                print(f"Try {d} at ({r},{c})")
            # place
            board[r][c] = d
            rows[r].add(d)
            cols[c].add(d)
            boxes[b_index].add(d)

            if backtrack(idx + 1):
                return True

            # undo
            if verbose:
                print(f"Backtrack {d} at ({r},{c})")
            board[r][c] = "."
            rows[r].remove(d)
            cols[c].remove(d)
            boxes[b_index].remove(d)

        return False

    return board if backtrack(0) else None
