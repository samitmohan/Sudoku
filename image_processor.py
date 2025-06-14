import streamlit as st
import cv2
import numpy as np
from typing import Tuple, List


def order_pts(pts: np.ndarray) -> np.ndarray:
    """Order 4 corner points as [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(d)]  # top-right
    rect[3] = pts[np.argmax(d)]  # bottom-left
    return rect


def detect_grid(image):
    """Locate largest grid in BGR image -> top down view -> return warped and top down view"""
    h, w = image.shape[:2]
    area = h * w

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)
    clean = cv2.bitwise_not(clean)

    # find contours and filter for large square
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    MIN_BOARD_AREA_RATIO = 0.05
    for c in contours:
        a = cv2.contourArea(c)
        if a < MIN_BOARD_AREA_RATIO * area:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        ar = cw / float(ch)
        if 0.8 <= ar <= 1.2:
            candidates.append(c)

    if not candidates:
        raise ValueError("No valid Sudoku grid found")

    # Approximate to 4 corners (largest quad)
    grid = max(candidates, key=cv2.contourArea)
    peri = cv2.arcLength(grid, True)
    approx = cv2.approxPolyDP(grid, 0.04 * peri, True)
    if len(approx) != 4:
        raise ValueError("Grid outline is not quadrilateral")

    pts = approx.reshape(4, 2)
    ordered = order_pts(pts)
    # perspective transform  : top down
    (tl, tr, br, bl) = ordered
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    side = int(max(widthA, widthB, heightA, heightB))

    dst = np.array(
        [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype="float32"
    )

    M = cv2.getPerspectiveTransform(ordered, dst)
    # warp to a flat square view
    warped = cv2.warpPerspective(image, M, (side, side))

    scale = 2  # Upsample factor
    warped_up = cv2.resize(
        warped, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )

    # Correct homography (original → warped_up)
    S_up = np.array(
        [[scale, 0, 0], [0, scale, 0], [0, 0, 1]],
        dtype=np.float32,
    )
    M_scaled = S_up @ M

    bw = cv2.cvtColor(warped_up, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(bw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cell = side // 9 * scale
    mask = np.ones_like(bw) * 255
    lines = cv2.HoughLinesP(
        bw,
        rho=1,
        theta=np.pi / 180,
        threshold=int(cell * 1.0),
        minLineLength=int(cell * 0.5),
        maxLineGap=10
    )
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), 0, thickness=7)
    return warped_up, M_scaled


def _font_params(cell_px: int) -> tuple[float, int]:
    """
    Return (font_scale, thickness) for cv2.putText, scaled
    to the side-length of a Sudoku cell.
    You can tweak the constants below until the look matches your template.
    """
    # these were tuned for ~40–70 px cells; they scale linearly:
    scale     = 1.0 * cell_px / 50     
    thickness = max(1, int(cell_px / 40))
    return scale, thickness

def draw_solution_overlay( original_img, warped_img, M, board, solved):
    side = warped_img.shape[0]
    cell = side // 9
    # start with a blank canvas so only the solved digits are pasted back
    scale, thick = _font_params(cell)
    overlay = np.zeros_like(warped_img)

    for row in range(9):
        for col in range(9):
            if board[row][col] == ".":
                digit = str(solved[row][col])
                x = col * cell + cell // 6
                y = row * cell + int(cell * 0.8)

                # y = row * cell + 3 * cell // 4
                cv2.putText(
                    overlay,
                    digit,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    (80, 80, 80),
                    thick,
                    cv2.LINE_AA,
                )

    # Warp overlay back to original perspective
    Minv = np.linalg.inv(M)
    h, w = original_img.shape[:2]
    result = cv2.warpPerspective(overlay, Minv, (w, h))

    gray_mask = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Combine with original image

    thick = cv2.dilate(gray_mask, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.threshold(thick, 5, 255, cv2.THRESH_BINARY)[1]

    inv_mask   = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(original_img, original_img, mask=inv_mask)
    foreground = cv2.bitwise_and(result, result, mask=mask)

    combined = cv2.add(background, foreground)
    return combined
