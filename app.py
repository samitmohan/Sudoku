import streamlit as st
from PIL import Image
import io
from model import process_sudoku_image
# Page config
st.set_page_config(page_title="Sudoku Solver", layout="centered")

# Title
st.markdown("<h1 style='text-align: center; color: green;'>Sudoku Solver</h1>", unsafe_allow_html=True)

# Upload / Scan buttons
col1, col2 = st.columns(2)
with col1:
    live_photo = st.camera_input("📷 Live Photo Scan")
    if st.button("📷 Live Photo Scan", use_container_width=True):
        uploaded_img = st.camera_input("Take a photo")
        
with col2:
    if st.button("📤 Upload Photo", use_container_width=True):
        uploaded_img = st.file_uploader("📤 Upload Photo", type=["png", "jpg", "jpeg"])

image_to_process = uploaded_img or live_photo
if image_to_process:
    st.image(image_to_process, caption="Input Sudoku", use_column_width=True)
    result_image = process_sudoku_image(image_to_process)
    st.image(result_image, caption="Solved Sudoku")
    # Convert result image to bytes for download
    if isinstance(result_image, Image.Image):
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        st.download_button(
            label="📥 Download Solved Sudoku",
            data=img_byte_arr,
            file_name="solved_sudoku.png",
            mime="image/png"
        )
else:
    st.write("Please upload image")


# Test : display image
if st.session_state.get("image_uploaded", False):
    st.image(st.session_state["image_uploaded"], use_column_width=True)
elif st.session_state.get("image_scanned", False):
    st.image(st.session_state["image_scanned"], use_column_width=True)

# Add spacer to push footer down
st.markdown("<div style='height: 300px;'></div>", unsafe_allow_html=True)

# Footer with center-aligned button
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
    show_rules = st.button("Rules")
    st.markdown('</div>', unsafe_allow_html=True)

# Show rules if button clicked
if show_rules:
    st.markdown("""
    ### 🧩 Sudoku Rules:
    1. Each row must contain the digits **1–9** without repeating.  
    2. Each column must contain the digits **1–9** without repeating.  
    3. Each 3×3 subgrid must contain the digits **1–9** without repeating.  
    4. No guessing! Every puzzle has only one solution using logic.
    """, unsafe_allow_html=True)
