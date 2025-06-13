# generate a synthetic 9×9 grid image, test extract_cells yields 81 crops of the right size

import numpy as np
import cv2
from image_processor import detect_grid, extract_cells, draw_solution_overlay


def gen_synthetic_grid_image():
    side = 9 * 28  
    img = np.zeros((side, side, 3), dtype=np.uint8)
    step = side // 9
    # draw heavier box lines every 3 cells
    for i in range(10):
        thickness = 2 if i % 3 == 0 else 1
        cv2.line(img, (i * step, 0), (i * step, side), (255, 255, 255), thickness)
        cv2.line(img, (0, i * step), (side, i * step), (255, 255, 255), thickness)
    return img


def test_extract_cells_and_sizes():
    warped = gen_synthetic_grid_image()
    cells = extract_cells(warped)
    assert len(cells) == 81
    # each cell should be 28×28 after preprocessing
    for cell in cells:
        assert isinstance(cell, np.ndarray)
        assert cell.shape == (28, 28)


def test_detect_and_overlay_roundtrip():
    # create a fake solved board (all dots, then fill center cell)
    img = gen_synthetic_grid_image()
    warped, M = detect_grid(img)
    # dummy boards
    orig = [["."] * 9 for _ in range(9)]
    sol  = [["."] * 9 for _ in range(9)]
    sol[4][4] = "5"  # put a digit in middle
    # overlay should not crash and should return same shape as img
    out = draw_solution_overlay(img, warped, M, orig, sol)
    assert out.shape == img.shape