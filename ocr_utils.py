import easyocr
import torch
import streamlit as st


@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(["en"], gpu=torch.cuda.is_available())


ocr_reader = get_ocr_reader()
