# assume image is here
'''
Preprocess Image & Extract Grid
Once image is loaded:
    Convert it to grayscale
    Find the largest square contour (the grid)
    Apply a perspective transform to get a top-down view
'''
from solver import Sudoku
import cv2
import numpy as np
from ocr import recognize_digit  # Optional
from image_utils import *  # Optional modularization

def process_sudoku_image(uploaded_file):
    # Step 1: Convert to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Step 2: Preprocess and extract grid
    warped = extract_grid(image)

    # Step 3: Slice into 81 cells
    cells = split_into_cells(warped)

    # Step 4: Recognize digits
    board = []
    for row in cells:
        board_row = []
        for cell_img in row:
            digit = recognize_digit(cell_img)
            board_row.append(digit if digit else ".")
        board.append(board_row)

    s = Sudoku()
    s.solveSudoku(board)

    # Step 6: Overlay or generate new image
    solved_img = overlay_solution(warped, board)

    return solved_img