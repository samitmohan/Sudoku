import streamlit as st
import numpy as np
import cv2
import config

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

from image_processor import detect_grid, extract_cells, draw_solution_overlay
from recognition import load_model 
from utils import probs_to_board
from sudoku_solver import solve
model = load_model(config.MODEL_PATH)

st.set_page_config(page_title="Sudoku Solver", layout="centered")
st.title("🧩 Sudoku Solver")


def solve_sudoku_frame(frame: np.ndarray) -> np.ndarray:
    warped, M = detect_grid(frame)
    cells = extract_cells(warped)
    probs = model.predict_batch(cells)
    board = probs_to_board(probs) # “unwraps” CNN’s 81 probability vectors into the 9×9 board format
    solved = solve(board)

    if solved is None:
        return frame

    return draw_solution_overlay(frame, warped, M, board, solved)


class SudokuTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        try:
            solved_rgb = solve_sudoku_frame(img_rgb)
            return cv2.cvtColor(solved_rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            return img_bgr


mode = st.sidebar.selectbox("Mode", ["Upload Image", "Live Camera"])

if mode == "Upload Image":
    st.markdown("### 📤 Upload a Sudoku photo")
    img_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if img_file is not None:
        # decode uploaded file to OpenCV BGR
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        solved = solve_sudoku_frame(rgb)
        st.image(solved, caption="Solved Sudoku", use_column_width=True)

else:
    st.markdown("### 🎥 Live Solver")
    webrtc_streamer(
        key="sudoku-live",
        mode="SENDRECV",
        video_transformer_factory=SudokuTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )

st.markdown("<div style='height: 300px;'></div>", unsafe_allow_html=True)

# Footer CSS
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
    st.markdown('</div>', unsafe_allow_html=True)

if show_rules:
    st.markdown("""
    ### 🧩 Rules
    1. Each row must contain the digits **1–9** without repeating  
    2. Each column must contain the digits **1–9** without repeating  
    3. Each 3×3 subgrid must contain the digits **1–9** without repeating  
    4. Every puzzle has a unique solution (no guessing required)
    """, unsafe_allow_html=True)