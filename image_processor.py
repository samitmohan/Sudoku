import cv2
import numpy as np
from typing import List, Tuple

def order_pts(pts: np.ndarray) -> np.ndarray:
    """ order 4 unordered corner points as: [top-left, top-right, bottom-right, bottom-left] """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]   # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]   # bottom-right has largest sum
    rect[1] = pts[np.argmin(diff)]# top-right has smallest diff
    rect[3] = pts[np.argmax(diff)]# bottom-left has largest diff
    return rect

def detect_grid(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect the largest 4-corner contour in the input RGB image,
    top-down square warp, return (warped, M), where
    M is the 3×3 perspective transform from input to warped.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 3)

    # adaptive threshold + invert
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    thresh = cv2.bitwise_not(thresh)

    # largest contour
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("No contours found for Sudoku grid.")
    largest = max(contours, key=cv2.contourArea)

    # approximate polygon & ensure 4 corners
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    if len(approx) != 4:
        raise ValueError("Could not find a 4-corner Sudoku grid.")
    pts = approx.reshape(4, 2)
    ordered = order_pts(pts)

    # warp size
    (tl, tr, br, bl) = ordered
    widthA  = np.linalg.norm(br - bl)
    widthB  = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    side = int(max(widthA, widthB, heightA, heightB))

    # destination contours
    dst = np.array([
        [0,      0],
        [side-1, 0],
        [side-1, side-1],
        [0,      side-1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, M, (side, side))
    return warped, M

def extract_cells(warped: np.ndarray) -> List[np.ndarray]:
    """ split warped board into 81 cells -> return list of 81 arrays """
    side = warped.shape[0]
    cell = side // 9
    margin = int(cell * 0.2)  

    cells = []
    for r in range(9):
        for c in range(9):
            x0 = c * cell + margin
            y0 = r * cell + margin
            x1 = (c + 1) * cell - margin
            y1 = (r + 1) * cell - margin

            crop = warped[y0:y1, x0:x1]
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

            # digit becomes white on black
            _, thresh = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )

            # pad to square then resize to 28×28
            h, w = thresh.shape
            max_dim = max(h, w)
            pad_vert = (max_dim - h) // 2
            pad_horz = (max_dim - w) // 2
            squared = cv2.copyMakeBorder(
                thresh,
                pad_vert, max_dim - h - pad_vert,
                pad_horz, max_dim - w - pad_horz,
                cv2.BORDER_CONSTANT,
                value=0
            )
            digit = cv2.resize(squared, (28, 28), interpolation=cv2.INTER_AREA)
            cells.append(digit)

    return cells

def draw_solution_overlay( frame: np.ndarray, warped: np.ndarray, M: np.ndarray, original_board: List[List[str]], solved_board: List[List[str]]) -> np.ndarray:
    """ overlay only the newly filled digits (where original_board[r][c]=='.') """
    overlay = np.zeros_like(warped)
    side = warped.shape[0]
    cell = side // 9

    # Draw text in green (R,G,B)
    for r in range(9):
        for c in range(9):
            if original_board[r][c] == ".":
                num = solved_board[r][c]
                # center-ish in each cell
                x = c * cell + cell // 2
                y = r * cell + int(cell * 0.75)
                cv2.putText(
                    overlay,
                    num,
                    (x - 10, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

    # warp the overlay back to original frame perspective
    invM = np.linalg.inv(M)
    h, w = frame.shape[:2]
    back = cv2.warpPerspective(overlay, invM, (w, h))

    # combine with original (70% original + 30% overlay)
    result = cv2.addWeighted(frame, 1.0, back, 0.7, 0)
    return result