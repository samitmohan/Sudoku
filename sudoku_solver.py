from collections import defaultdict


class Sudoku:
    def solveSudoku(self, board):
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]
        empty_cells = []
      
        for i in range(9):
            for j in range(9):
                val = board[i][j]
                if val == '.': empty_cells.append((i, j))
                else:
                    rows[i].add(val)
                    cols[j].add(val)
                    boxes[(i//3)*3+(j//3)].add(val)

        def backtrack(index):
            if index == len(empty_cells): return True

            r,c = empty_cells[index]
            b = (r//3)*3+(c//3)
            for num in map(str, range(1, 10)):
                if num not in rows[r] and num not in cols[c] and num not in boxes[b]:
                    board[r][c] = num
                    rows[r].add(num)
                    cols[c].add(num)
                    boxes[b].add(num)

                    if backtrack(index + 1): return True

                    board[r][c] = '.'
                    rows[r].remove(num)
                    cols[c].remove(num)
                    boxes[b].remove(num)
            return False

        backtrack(0)
        return board

# need to input this board from camera 
board=[
    ["5", "3", ".", ".", "7", ".", ".", ".", "."],
    ["6", ".", ".", "1", "9", "5", ".", ".", "."],
    [".", "9", "8", ".", ".", ".", ".", "6", "."],
    ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
    ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
    ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
    [".", "6", ".", ".", ".", ".", "2", "8", "."],
    [".", ".", ".", "4", "1", "9", ".", ".", "5"],
    [".", ".", ".", ".", "8", ".", ".", "7", "9"],
]
s = Sudoku()
s.solveSudoku(board)
for row in board:
    print(row)