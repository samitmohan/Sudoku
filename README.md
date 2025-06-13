Program which solves sudoku (live and image upload)


```bash
source /venv/bin/activate
pip3 install -r requirements.txt or use uv
```

TODO: Try to use minitorch for CNN, also Yolo

sudoku‐cnn‐solver/
│
├── README.md               
├── requirements.txt        
└── config.py               # global constants: image sizes, paths, thresholds
│
├── data/                   
│   ├── sudoku_samples.json # a few puzzles + solutions for quick tests
│   └── mnist/              
│
├── models/
│   └── digit_recognizer.pth  
│
├── app.py                  
│
├── sudoku_solver.py        # backtracking solver: solve(board)->solved_board
│
├── image_processor.py      # classical CV to find & warp grid, crop 81 cells:
│   ├── detect_grid(img)         
│   ├── extract_cells(warped)    
│   └── draw_solution_overlay() 
│
├── recognition.py          # CNN digit‐classifier + inference wrappers:
│   ├── class DigitClassifier(nn.Module)
│   ├── load_model(path)->model
│   └── predict_batch(cell_images)->[81×10] probabilities
│
├── train_digit_model.py    # trains DigitClassifier on MNIST, saves .pth
│
├── utils.py                # helpers:
│   ├── show_board(board)  
│   └── load_sudoku_samples(...)
│
└── tests/                  
    ├── test_solver.py
    ├── test_image_processor.py
    └── test_recognition.py