import streamlit as st

st.set_page_config(page_title="🧩 AI Sudoku Solver", layout="centered")
import numpy as np
import cv2
import config
from sudoku_solver import solve
import torch
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from image_processor import detect_grid, draw_solution_overlay
from recognition import load_model, easyocr_digit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def get_model():
    m = load_model(config.MODEL_PATH)
    return m.to(DEVICE).eval()


def solve_sudoku_frame(bgr: np.ndarray) -> np.ndarray:
    model = get_model()
    # warp
    warped_up, M = detect_grid(bgr)
    side = warped_up.shape[0]
    cell = side // 9

    # threshold & remove grid lines via Hough
    gray = cv2.cvtColor(warped_up, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # build mask and use HoughLinesP to knock out straight grid lines
    mask = np.ones_like(bw) * 255
    lines = cv2.HoughLinesP(
    bw,
    rho=1,
    theta=np.pi / 180,
    threshold=int(cell * 1.2),      # ← was 1.5
    minLineLength=int(cell * 0.6),  # ← was 0.8
    maxLineGap=8                    # ← was 5
    )
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(mask, (x1, y1), (x2, y2), 0, thickness=5)    

    # subtract only those lines, leaving digit blobs
    digits_only = cv2.bitwise_and(bw, mask)

    # each digit contour
    digits_only = cv2.morphologyEx(digits_only, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    cnts, _ = cv2.findContours(digits_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    board = [["." for _ in range(9)] for __ in range(9)]

    rois = []  # batches
    positions = []
    
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # skip too-small blobs
        if w * h < (cell * cell) * 0.05:
            continue

        # stay within image bounds
        pad = 5
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(side, x + w + pad)
        y1 = min(side, y + h + pad)
        roi = digits_only[y0:y1, x0:x1]

        # skip empty or too-small regions
        if roi.size == 0 or roi.shape[0] < 2 or roi.shape[1] < 2:
            continue

        # resize to 28×28 for classification
        roi28 = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi28 = cv2.GaussianBlur(roi28, (3, 3), 0)
        roi28 = cv2.threshold(roi28, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # map back to grid
        cx, cy = x + w // 2, y + h // 2
        col, row = min(cx // cell, 8), min(cy // cell, 8)
        rois.append(roi28)
        positions.append((row, col))

    # 4) Classify all ROIs in one batch (CNN)
    if rois:
        for (row, col), roi28 in zip(positions, rois):
            # use model to predict the digit
            input_tensor = (
                torch.tensor(roi28, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                / 255.0
            )
            # input_tensor = (input_tensor - 0.1307) / 0.3081
            input_tensor = input_tensor.to(DEVICE)
            with torch.no_grad():
                output = model(input_tensor)
                pred = output.argmax(dim=1).item()
                softmax_probs = torch.softmax(output, dim=1)[0]
                confidence = softmax_probs[pred].item()

            if confidence < 0.90 or pred == 0:
                pred = easyocr_digit(roi28)  # fallback
            if pred != 0:
                if 0 <= row < 9 and 0 <= col < 9:
                    board[row][col] = str(pred)
    else:
        st.warning("No digit blobs detected – check your grid extraction.")

    debug = st.sidebar.checkbox("🔍 Show Debug Info", value=False)

    orig_board = [row[:] for row in board]        # shallow copy of the 9×9 list

    solved = solve(board)                         # board will be mutated here

    if isinstance(solved, str):
        st.warning(f"⚠️ Invalid puzzle: {solved}")
        if debug:
            st.write(board)
        return bgr
    if debug:
        st.sidebar.image(bw, caption="BW after Otsu")
        st.sidebar.image(mask, caption="Hough mask")
        st.sidebar.image(digits_only, caption="digits_only")
        st.write("🧩 Input Board", board)
        st.write("✅ Solved Board", solved)
        st.sidebar.image(
            cv2.cvtColor(warped_up, cv2.COLOR_BGR2RGB),
            caption="Warped",
            use_container_width=True,
        )

    if solved:
        return draw_solution_overlay(
            bgr,
            warped_up,
            M,
            orig_board,  
            solved,
        )
    else:
        st.warning("⚠️ No solution found")
        return bgr


class SudokuTransformer(VideoTransformerBase):
    def transform(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        try:
            return solve_sudoku_frame(img_bgr)
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
            return img_bgr


mode = st.sidebar.selectbox("Mode", ["Upload Image", "Live Camera"])

if mode == "Upload Image":
    st.markdown("### 📤 Upload a Sudoku photo")
    f = st.file_uploader("Choose image...", type=["png", "jpg", "jpeg"])
    if f:
        data = np.frombuffer(f.read(), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        try:
            solved_bgr = solve_sudoku_frame(bgr)
            st.image(cv2.cvtColor(solved_bgr, cv2.COLOR_BGR2RGB), caption="Solved")
        except Exception as e:
            st.warning(f"⚠️ {e}")

else:
    st.markdown("### 🎥 Live Sudoku Solver")
    webrtc_streamer(
        key="sudoku-live",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=SudokuTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )

st.markdown("<div style='height: 300px;'></div>", unsafe_allow_html=True)
footer_css = """
<style>
.footer {
    position: relative;
    bottom: 0;
    width: 100%;
    text-align: center;
}
</style>
"""
st.markdown(footer_css, unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    show_rules = st.button("📝 Rules")
    st.markdown("</div>", unsafe_allow_html=True)

if show_rules:
    st.markdown(
        """
    ### 🧩 Rules
    1. Each row must contain the digits **1–9** without repeating  
    2. Each column must contain the digits **1–9** without repeating  
    3. Each 3×3 subgrid must contain the digits **1–9** without repeating  
    4. Every puzzle has a unique solution (no guessing required)
    """,
        unsafe_allow_html=True,
    )
