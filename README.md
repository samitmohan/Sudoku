# 🧠 AI Sudoku Solver (Web App)

A real-time AI-powered Sudoku Solver that lets you scan a printed or handwritten Sudoku puzzle (e.g., from a newspaper) using your **laptop or phone camera**, and instantly solves it with a clean visual overlay.

Built using **Python**, **OpenCV**, **Tesseract OCR** / **CNN MNIST**, and **Streamlit** for a smooth, web-based experience — no installations required to use!

---

## 🚀 Features

- 📷 Capture Sudoku from your device camera
- 🧠 Automatically detects the grid and extracts digits
- 🔍 Uses **OCR (Tesseract)** or **CNN** for digit recognition
- ✨ Solves the puzzle in real-time using a backtracking algorithm
- 🖼️ Overlays the solution on the original scanned image
- 🌐 Works on laptops and phones via browser

---

## 📸 Demo

> live link

---

## 🧱 Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **Image Processing**: OpenCV
- **Digit Recognition**:
  - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
  - (Optional: CNN trained on MNIST or custom digits)
- **Sudoku Solver**: Python backtracking algorithm

---

## 🖥️ Usage

- Launch the app in your browser.
- Click "Scan Sudoku from Camera" and capture a clear photo.
- Let the AI extract, analyze, and solve the puzzle.
- View the solved grid overlay on your original image.

## Project Structure

📁 sudoku-solver-streamlit/
    ├── app.py # Streamlit UI and app logic
    ├── solver.py # Sudoku solving algorithm
    ├── ocr.py # Digit recognition (Tesseract/CNN)
    ├── utils.py # Helper functions (grid detection, warping)
    ├── requirements.txt # Python dependencies
    └── README.md # Project documentation

# TODO

- Manual correction option before solving
